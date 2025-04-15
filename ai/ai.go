package ai

import (
	"context"
	"fmt"
	"io"
	"strings"

	"github.com/rjanupam/aiterm/config"
)

type AIProvider interface {
	ProcessPrompt(ctx context.Context, prompt string, writer io.Writer) (string, error)
	Close() error
	ClearHistory() error
}

type AIClient struct {
	provider AIProvider
	config   *config.Config
}

func NewAIClient(ctx context.Context, modelName string, cfg *config.Config) (*AIClient, error) {
	var provider AIProvider
	var err error

	if modelName == "" {
		modelName = cfg.Model
	}

	switch {
	case strings.HasPrefix(strings.ToLower(modelName), "gemini"):
		provider, err = GeminiClient(ctx, modelName, cfg)
	// case strings.HasPrefix(strings.ToLower(modelName), "xai"):
	// 	provider, err = NewXAIClient(ctx, modelName, cfg)
	default:
		return nil, fmt.Errorf("unsupported model: %s", modelName)
	}

	if err != nil {
		return nil, fmt.Errorf("failed to initialize provider: %v", err)
	}

	return &AIClient{
		provider: provider,
		config:   cfg,
	}, nil
}

func (ai *AIClient) ProcessPrompt(ctx context.Context, prompt string, writer io.Writer) (string, error) {
	return ai.provider.ProcessPrompt(ctx, prompt, writer)
}

func (ai *AIClient) ClearHistory() error {
	return ai.provider.ClearHistory()
}

func (ai *AIClient) Close() error {
	return ai.provider.Close()
}
