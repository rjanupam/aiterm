package ai

import (
	"context"
	"fmt"
	"io"
	"os"
	"strings"

	"google.golang.org/api/option"
	"github.com/google/generative-ai-go/genai"
	"github.com/rjanupam/aiterm/config"
)

type geminiClient struct {
	client *genai.Client
	chat   *genai.ChatSession
	config *config.Config
}

func GeminiClient(ctx context.Context, modelName string, cfg *config.Config) (AIProvider, error) {
	apiKey := cfg.GetAPIKey("gemini")
	if apiKey == "" {
		apiKey = os.Getenv("GEMINI_API_KEY")
		if apiKey == "" {
			return nil, fmt.Errorf("GEMINI_API_KEY not found in config or environment")
		}
	}

	client, err := genai.NewClient(ctx, option.WithAPIKey(apiKey))
	if err != nil {
		return nil, fmt.Errorf("failed to create Gemini client: %v", err)
	}

	model := client.GenerativeModel(modelName)
	if model == nil {
		client.Close()
		return nil, fmt.Errorf("failed to initialize model: %s", modelName)
	}

	model.SetMaxOutputTokens(int32(cfg.MaxTokens))
	model.SetTemperature(float32(cfg.Temperature))

	systemPrompt := GetSystemPrompt()

	chat := model.StartChat()
	chat.History = append(chat.History, &genai.Content{
		Parts: []genai.Part{genai.Text(systemPrompt)},
		Role:  "model",
	})

	return &geminiClient{
		client: client,
		chat:   chat,
		config: cfg,
	}, nil
}

func (g *geminiClient) ProcessPrompt(ctx context.Context, prompt string, writer io.Writer) (string, error) {
	var fullResponse strings.Builder
	stream := g.chat.SendMessageStream(ctx, genai.Text(prompt))
	
	for {
		resp, err := stream.Next()
		if err != nil {
			if err.Error() == "no more items in iterator" {
				break
			}
			fmt.Fprintf(os.Stderr, "\nStreaming error: %v, falling back to non-streaming\n", err)
			return g.processNonStreaming(ctx, prompt, writer)
		}

		if len(resp.Candidates) > 0 {
			for _, part := range resp.Candidates[0].Content.Parts {
				if text, ok := part.(genai.Text); ok {
					fmt.Fprint(writer, string(text))
					fullResponse.WriteString(string(text))
				}
			}
		}
	}
	fmt.Fprintln(writer)

	g.chat.History = append(g.chat.History, &genai.Content{
		Parts: []genai.Part{genai.Text(prompt)},
		Role:  "user",
	})
	if fullResponse.Len() > 0 {
		g.chat.History = append(g.chat.History, &genai.Content{
			Parts: []genai.Part{genai.Text(fullResponse.String())},
			Role:  "model",
		})
	}

	return fullResponse.String(), nil
}

func (g *geminiClient) processNonStreaming(ctx context.Context, prompt string, writer io.Writer) (string, error) {
	resp, err := g.chat.SendMessage(ctx, genai.Text(prompt))
	if err != nil {
		return "", fmt.Errorf("failed to generate content: %v", err)
	}

	if len(resp.Candidates) == 0 {
		return "", fmt.Errorf("no response candidates returned")
	}

	var fullResponse strings.Builder
	for _, part := range resp.Candidates[0].Content.Parts {
		if text, ok := part.(genai.Text); ok {
			fmt.Fprint(writer, string(text))
			fullResponse.WriteString(string(text))
		}
	}
	fmt.Fprintln(writer)

	g.chat.History = append(g.chat.History, &genai.Content{
		Parts: []genai.Part{genai.Text(prompt)},
		Role:  "user",
	})
	if fullResponse.Len() > 0 {
		g.chat.History = append(g.chat.History, &genai.Content{
			Parts: []genai.Part{genai.Text(fullResponse.String())},
			Role:  "model",
		})
	}

	return fullResponse.String(), nil
}

func (g *geminiClient) ClearHistory() error {
	if len(g.chat.History) > 0 && g.chat.History[0].Role == "model" {
		g.chat.History = g.chat.History[:1]
	} else {
		g.chat.History = []*genai.Content{}
	}
	fmt.Println("Gemini history cleared (keeping system prompt if applicable).")
	return nil
}

func (g *geminiClient) Close() error {
	if g.client != nil {
		return g.client.Close()
	}
	return nil
}
