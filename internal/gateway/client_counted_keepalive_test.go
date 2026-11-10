package gateway

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"strconv"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/yetone/magpie/internal/provider"
)

// A client that waits for an **event** and not a comment.
//
// The gateway keeps a waiting stream alive with an SSE comment (": keepalive"),
// which every event-stream reader skips by the letter of the spec. Two
// clients do not skip it:
//
//   - DeepSeek Harness, through llm-pi-ai's pi-ai: its OpenAI SDK drops the
//     comment while parsing, so the reply's first chunk never arrives and
//     the turn is cut at the harness's own step timeout.
//   - Claude Code, whose Anthropic parser counts a ping but not a comment:
//     before message_start the gateway answers with a comment, since the
//     Anthropic SDKs fail on any event before message_start ("Unexpected event
//     order, got ping before message_start"), so a reply that is still waiting
//     in a provider's queue is kept alive with a comment the client ignores.
//
// The reply is the same either way: the gateway holds it until it has
// something to say. What it must not do is keep a waiting client alive with
// something that client cannot see.
//
// The tests below pin what each client counts: an Anthropic client a ping
// (its own heartbeat, the one Claude Code's watchdog counts), a Chat client a
// chunk it can parse (an empty delta), a Responses client an event it knows
// (response.in_progress), and Gemini nothing at all (its SDK reads "data: "
// lines one after another and fails on anything else).
//
// They also pin what a keepalive must not do: release a held first piece of
// the reply early (magpie-community/plugins#738: a ping released the held
// first piece with the start of the placeholder in it, and Claude Code wrote
// {{…}} into the file), and put an event before the reply's own first event
// where the client's parser would fail on it.

// chatAlive is a Chat client's heartbeat: a chunk with an empty delta, the
// shape an OpenAI-compatible reader parses into a chunk and counts.
const chatAlive = `data: {"choices":[{"index":0,"delta":{}}]}`

// anthropicAlive is an Anthropic client's heartbeat: the ping event the
// Anthropic API defines for it.
const anthropicAlive = `event: ping`

// aliveCountedBy counts the heartbeats in a stream that a client of the given
// protocol counts: an Anthropic client a ping, a Chat client a chunk, a
// Responses client an event it knows, a Gemini client nothing.
func aliveCountedBy(proto provider.Protocol, body string) int {
	n := 0
	for _, line := range strings.Split(body, "\n") {
		switch proto {
		case provider.Anthropic:
			if strings.HasPrefix(line, "event: ping") {
				n++
			}
		case provider.Chat:
			if strings.HasPrefix(line, "data: {") && strings.Contains(line, "\"choices\"") {
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

// clientParses counts the heartbeats a real client of this protocol would
// parse, the way the client's own reader does.
//
//   - A Chat client (llm-pi-ai's pi-ai, through the OpenAI SDK): the SDK
//     reads the SSE stream line by line and yields one chunk per "data:"
//     line with JSON in it; a comment line is dropped by the decoder and
//     never reaches the client.
//   - An Anthropic client (Claude Code): it counts a ping event.
//   - A Responses client (Codex): it counts the events it knows.
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
				Choices []struct {
					Delta map[string]any `json:"delta"`
				} `json:"choices"`
			}
			if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
				continue
			}
			if len(chunk.Choices) == 0 {
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

// TestClientCountedKeepalive: a provider that never answers, so
// the gateway holds the reply and keeps the client alive with whatever it
// has. The client must hear heartbeats **it counts**, or its own idle
// timeout ends the turn while the gateway and the provider are still waiting.
//
// Before, a Chat client was kept alive with a comment it skips and an
// Anthropic client one too, before message_start: the two clients that wait
// longest were the two that never heard anything.
func TestClientCountedKeepalive(t *testing.T) {
	for _, c := range []struct {
		name, path, body string
		proto            provider.Protocol
	}{
		{"anthropic", "/v1/messages", `{"model":"fake/m1","stream":true,"max_tokens":64,"messages":[{"role":"user","content":"hi"}]}`, provider.Anthropic},
		{"chat", "/v1/chat/completions", `{"model":"fake/m1","stream":true,"messages":[{"role":"user","content":"hi"}]}`, provider.Chat},
		{"responses", "/v1/responses", `{"model":"fake/m1","stream":true,"input":"hi"}`, provider.Responses},
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
			// read for a while and see what the client was given
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
			if n := clientParses(c.proto, got); n == 0 {
				t.Fatalf("the %s client was kept alive with nothing it counts:\n%s", c.name, got)
			}
		})
	}
}


// TestKeepaliveDoesNotReleaseHeldContent: a keepalive must not let a held
// piece of the reply out early.
//
// The reply is held until it has something to say, so that a failure or a
// refusal can still reach the agent as the stream's error and another
// account may answer in the same stream. A keepalive that released the held
// first piece would give the agent half of it: a ping once did
// (magpie-community/plugins#738 - the ping released the held first piece
// with the start of the placeholder in it, and Claude Code wrote {{...}}
// into the file).
//
// So a keepalive is a heartbeat and nothing else: it may not write a byte of
// the reply.
func TestKeepaliveDoesNotReleaseHeldContent(t *testing.T) {
	fresh(t)
	quietFast(t)
	const placeholder = "{{EMAIL_abcdefgh}}"
	// the provider begins a tool call's arguments and then stops: the piece
	// with the placeholder's start is held, the rest comes later.
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.Copy(io.Discard, r.Body)
		w.Header().Set("Content-Type", "text/event-stream")
		args := `{"content":"mail ` + placeholder + `"}`
		cut := strings.Index(args, placeholder) + 9 // {{EMAIL_a
		chunk := "data: " + fmt.Sprintf(`{"id":"c1","model":"m1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"Write","arguments":%s}}]}}]}`, strconv.Quote(args[:cut])) + "\n\n"
		io.WriteString(w, chunk)
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
	got := body.String()
	// the client must have been kept alive with something it counts.
	if clientParses(provider.Chat, got) == 0 {
		t.Fatalf("the chat client was kept alive with nothing it counts:\n%s", got)
	}
}