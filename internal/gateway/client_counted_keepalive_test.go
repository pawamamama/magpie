package gateway

import (
	"bufio"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/yetone/magpie/internal/provider"
)

// A waiting stream is kept alive with the SSE comment ": keepalive", which
// every event-stream reader skips by the letter of the spec. One client does
// not skip it, and two must not be handed an event before their reply begins.
//
//   - DeepSeek Harness, through llm-pi-ai's pi-ai: its OpenAI SDK's SSE
//     decoder drops the comment, so the reply's first chunk never arrives and
//     the harness's own step timeout ends the turn while the provider works.
//     It is kept alive with a chunk it parses — an empty delta, a chunk and
//     not a word.
//   - Anthropic: the SDKs' MessageStream fails on any event before
//     message_start ("Unexpected event order, got ping before message_start").
//     A held reply has not started, so it is kept with the comment, never a
//     ping (anthropicEncoder.keepalive answers a comment there for the same
//     reason).
//   - Responses: its events carry a sequence_number the encoder owns
//     (responsesEncoder.write); one written raw would be numbered out of step
//     with the encoder's own stream. Codex is met by agentQuietMost instead.
//
// These tests pin which client is kept alive with what, and that a heartbeat
// is a heartbeat and nothing else: it may not release a held piece of the
// reply early (magpie-community/plugins#738 — a ping released the held first
// piece with the start of the placeholder in it, and Claude Code wrote {{…}}
// into the file; TestMaskedToolArgsSplitAroundPing covers that path).

// clientParses counts the heartbeats a real client of this protocol would
// parse, the way the client's own reader does.
//
//   - A Chat client (llm-pi-ai's pi-ai, through the OpenAI SDK): the SDK
//     reads the SSE stream line by line and yields one chunk per "data:" line
//     with JSON in it; a comment line is dropped by the decoder and never
//     reaches the client.
//   - An Anthropic client: it counts a ping event.
//   - A Responses client: it counts the events it knows.
func clientParses(proto provider.Protocol, body string) int {
	n := 0
	for _, line := range strings.Split(body, "\n") {
		if strings.HasPrefix(line, ":") {
			continue // an SSE comment: every reader skips it
		}
		switch proto {
		case provider.Chat:
			if !strings.HasPrefix(line, "data:") {
				continue
			}
			payload := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
			if payload == "" || payload == "[DONE]" {
				continue
			}
			var chunk struct {
				Object  string `json:"object"`
				Choices []struct {
					Delta map[string]any `json:"delta"`
				} `json:"choices"`
			}
			if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
				continue
			}
			// a Chat client parses a chunk, which the API marks
			// "chat.completion.chunk" and gives a choices array.
			if chunk.Object != "chat.completion.chunk" || len(chunk.Choices) == 0 {
				continue
			}
			n++
		case provider.Anthropic:
			if strings.HasPrefix(line, "event: ping") {
				n++
			}
		case provider.Responses:
			if strings.HasPrefix(line, "event: response.") {
				n++
			}
		}
	}
	return n
}

// TestClientCountedKeepalive: a provider that never answers, so the gateway
// holds the reply and keeps the client alive with whatever it has. A Chat
// client must hear a chunk it parses; an Anthropic or a Responses client must
// hear nothing but the comment, since any event before its reply begins breaks
// its parser.
func TestClientCountedKeepalive(t *testing.T) {
	for _, c := range []struct {
		name, path, body string
		proto            provider.Protocol
		wantsEvent       bool // kept alive with a protocol event here
	}{
		{"chat", "/v1/chat/completions", `{"model":"fake/m1","stream":true,"messages":[{"role":"user","content":"hi"}]}`, provider.Chat, true},
		{"anthropic", "/v1/messages", `{"model":"fake/m1","stream":true,"max_tokens":64,"messages":[{"role":"user","content":"hi"}]}`, provider.Anthropic, false},
		{"responses", "/v1/responses", `{"model":"fake/m1","stream":true,"input":"hi"}`, provider.Responses, false},
	} {
		t.Run(c.name, func(t *testing.T) {
			fresh(t)
			quietFast(t)
			// the provider says nothing at all: the gateway holds the reply
			// and keeps the client alive on its own.
			up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				io.Copy(io.Discard, r.Body)
				w.Header().Set("Content-Type", "text/event-stream")
				w.(http.Flusher).Flush()
				<-r.Context().Done()
			}))
			defer up.Close()
			p := provider.Provider{ID: "fake", Name: "Fake", Key: "fixture", Models: []string{"m1"}}
			p.Chat = up.URL + "/v1"
			p.Responses = up.URL + "/v1"
			if err := provider.Save(p); err != nil {
				t.Fatal(err)
			}
			server := httptest.NewServer(New().Handler())
			defer server.Close()
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			req, _ := http.NewRequestWithContext(ctx, "POST", server.URL+c.path, strings.NewReader(c.body))
			req.Header.Set("Content-Type", "application/json")
			res, err := http.DefaultClient.Do(req)
			if err != nil {
				t.Fatal(err)
			}
			defer res.Body.Close()
			if res.StatusCode != http.StatusOK {
				t.Fatalf("status %d", res.StatusCode)
			}
			var body strings.Builder
			sc := bufio.NewScanner(res.Body)
			done := make(chan struct{})
			go func() {
				defer close(done)
				for sc.Scan() {
					body.WriteString(sc.Text())
					body.WriteString("\n")
				}
			}()
			select {
			case <-done:
			case <-time.After(600 * time.Millisecond):
			}
			cancel()
			got := body.String()
			n := clientParses(c.proto, got)
			if c.wantsEvent {
				if n == 0 {
					t.Fatalf("the %s client was kept alive with nothing it counts:\n%s", c.name, got)
				}
			} else if n != 0 {
				t.Fatalf("the %s client was sent an event before its reply began, where its parser fails on it:\n%s", c.name, got)
			}
		})
	}
}

// TestHeartbeatCarriesNoWord: a heartbeat is a heartbeat and nothing else.
// The Chat keepalive is a chunk, but its delta carries no content, no
// reasoning and no call — a client shows nothing for it. The reply is held and
// released by the provider's own bytes, never by a heartbeat
// (magpie-community/plugins#738, the Anthropic path, is
// TestMaskedToolArgsSplitAroundPing).
func TestHeartbeatCarriesNoWord(t *testing.T) {
	fresh(t)
	quietFast(t)
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.Copy(io.Discard, r.Body)
		w.Header().Set("Content-Type", "text/event-stream")
		w.(http.Flusher).Flush()
		<-r.Context().Done()
	}))
	defer up.Close()
	p := provider.Provider{ID: "fake", Name: "Fake", Key: "fixture", Models: []string{"m1"}}
	p.Chat = up.URL + "/v1"
	if err := provider.Save(p); err != nil {
		t.Fatal(err)
	}
	server := httptest.NewServer(New().Handler())
	defer server.Close()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	req, _ := http.NewRequestWithContext(ctx, "POST", server.URL+"/v1/chat/completions",
		strings.NewReader(`{"model":"fake/m1","stream":true,"messages":[{"role":"user","content":"hi"}]}`))
	req.Header.Set("Content-Type", "application/json")
	res, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()
	var body strings.Builder
	sc := bufio.NewScanner(res.Body)
	done := make(chan struct{})
	go func() {
		defer close(done)
		for sc.Scan() {
			body.WriteString(sc.Text())
			body.WriteString("\n")
		}
	}()
	select {
	case <-done:
	case <-time.After(600 * time.Millisecond):
	}
	cancel()
	seen := 0
	for _, line := range strings.Split(body.String(), "\n") {
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		payload := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if payload == "" || payload == "[DONE]" {
			continue
		}
		var chunk struct {
			Choices []struct {
				Delta map[string]any `json:"delta"`
			} `json:"choices"`
		}
		if json.Unmarshal([]byte(payload), &chunk) != nil || len(chunk.Choices) == 0 {
			continue
		}
		seen++
		d := chunk.Choices[0].Delta
		for _, k := range []string{"content", "reasoning_content", "reasoning", "tool_calls"} {
			if v, ok := d[k]; ok && v != nil && v != "" {
				t.Fatalf("a heartbeat carried %q=%v:\n%s", k, v, body.String())
			}
		}
	}
	if seen == 0 {
		t.Fatalf("no heartbeat chunk was parsed:\n%s", body.String())
	}
}
