package ai

import (
	"context"
	"fmt"
	"os"
	"strings"

	"github.com/google/generative-ai-go/genai"
	"google.golang.org/api/option"
)

type AIClient struct {
	client  *genai.Client
	chat    *genai.ChatSession
	history []*genai.Content
}

func NewAIClient(ctx context.Context, modelName string) (*AIClient, error) {
	apiKey := os.Getenv("GEMINI_API_KEY")
	if apiKey == "" {
		return nil, fmt.Errorf("GEMINI_API_KEY environment variable is not set")
	}

	client, err := genai.NewClient(ctx, option.WithAPIKey(apiKey))
	if err != nil {
		return nil, fmt.Errorf("failed to create AI client: %w", err)
	}

	model := client.GenerativeModel(modelName)
	if model == nil {
		client.Close()
		return nil, fmt.Errorf("failed to initialize model: %s", modelName)
	}

	systemPrompt := `You are a Linux command-line assistant. When asked to provide shell commands:
1. Provide a brief explanation of what the commands do.
2. Include exactly one markdown code block containing all commands or scripts, if any.
3. Ensure the code block is Linux-compatible and executable as a single bash script.
4. Avoid dangerous commands (e.g., rm -rf /) unless explicitly requested with clear intent.
5. For non-command requests, provide a clear, concise answer without code blocks.
Keep responses short, practical, and limited to one code block for commands.`

	chat := model.StartChat()
	chat.History = append(chat.History, &genai.Content{
		Parts: []genai.Part{genai.Text(systemPrompt)},
		Role:  "model",
	})

	return &AIClient{
		client:  client,
		chat:    chat,
		history: []*genai.Content{},
	}, nil
}

func (ai *AIClient) ProcessPrompt(ctx context.Context, prompt string) (string, error) {
	fmt.Print("AI: ")

	var fullResponse strings.Builder
	stream := ai.chat.SendMessageStream(ctx, genai.Text(prompt))
	for {
		resp, err := stream.Next()
		if err != nil {
			if err.Error() == "no more items in iterator" {
				break
			}
			fmt.Fprintf(os.Stderr, "\nStreaming error: %v, falling back to non-streaming\n", err)
			return fullResponse.String(), fmt.Errorf("streaming error")
		}

		if len(resp.Candidates) > 0 {
			for _, part := range resp.Candidates[0].Content.Parts {
				if text, ok := part.(genai.Text); ok {
					fmt.Print(string(text))
					fullResponse.WriteString(string(text))
				}
			}
		}
	}
	fmt.Println()

	ai.history = append(ai.history, &genai.Content{
		Parts: []genai.Part{genai.Text(prompt)},
		Role:  "user",
	})
	if fullResponse.Len() > 0 {
		ai.history = append(ai.history, &genai.Content{
			Parts: []genai.Part{genai.Text(fullResponse.String())},
			Role:  "model",
		})
	}

	return fullResponse.String(), nil
}

func (ai *AIClient) GetHistory() []*genai.Content {
	return ai.history
}

func (ai *AIClient) ClearHistory() {
	ai.history = []*genai.Content{}
	ai.chat.History = []*genai.Content{}
}

func (ai *AIClient) Close() error {
	if ai.client != nil {
		return ai.client.Close()
	}
	return nil
}
