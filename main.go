package main

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/rjanupam/aiterm/ai"
)

func main() {
	ctx := context.Background()

	aiClient, err := ai.NewAIClient(ctx, "gemini-1.5-flash")
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to initialize AI client: %v\n", err)
		os.Exit(1)
	}
	defer aiClient.Close()

	reader := bufio.NewReader(os.Stdin)

	for {
		fmt.Print("aiterm> ")
		input, err := reader.ReadString('\n')
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error reading input: %v\n", err)
			continue
		}

		input = strings.TrimSpace(input)
		if input == "exit" {
			break
		}
		if input == "clear" {
			aiClient.ClearHistory()
			fmt.Println("Conversation history cleared.")
			continue
		}

		if strings.HasPrefix(input, "$") {
			command := strings.TrimSpace(input[1:])
			if command == "" {
				continue
			}
			if err := executeDirectCommand(command); err != nil {
				fmt.Fprintf(os.Stderr, "Error executing command: %v\n", err)
			}
			continue
		}

		response, err := aiClient.ProcessPrompt(ctx, input)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error processing prompt: %v\n", err)
			continue
		}

		codeBlock := extractBashCodeBlock(response)
		if codeBlock != "" {
			tmpFile, err := saveToTempFile(codeBlock)
			if err != nil {
				fmt.Fprintf(os.Stderr, "Error saving script: %v\n", err)
				continue
			}
			defer os.Remove(tmpFile)

			content, err := os.ReadFile(tmpFile)
			if err != nil {
				fmt.Fprintf(os.Stderr, "Error reading script: %v\n", err)
				continue
			}
			fmt.Println("--------------------------------------------------")
			fmt.Print(string(content))
			fmt.Println("--------------------------------------------------")

			fmt.Print("Execute this script? (Y/n): ")
			confirm, _ := reader.ReadString('\n')
			confirm = strings.TrimSpace(strings.ToLower(confirm))
			if confirm == "" {
				confirm = "y"
			}

			if confirm == "y" {
				if err := executeScript(tmpFile); err != nil {
					fmt.Fprintf(os.Stderr, "Error executing script: %v\n", err)
				}
			} else {
				fmt.Println("Script not executed.")
			}
		}
	}
}

func extractBashCodeBlock(response string) string {
	re := regexp.MustCompile("(?s)```bash\n(.*?)\n```")
	match := re.FindStringSubmatch(response)
	if len(match) > 1 {
		return strings.TrimSpace(match[1])
	}
	return ""
}

func saveToTempFile(code string) (string, error) {
	tmpDir := os.TempDir()
	tmpFile := filepath.Join(tmpDir, fmt.Sprintf("aiterm-%d.sh", os.Getpid()))
	f, err := os.Create(tmpFile)
	if err != nil {
		return "", err
	}
	defer f.Close()

	if _, err := f.WriteString("#!/bin/bash\n" + code + "\n"); err != nil {
		return "", err
	}

	if err := os.Chmod(tmpFile, 0700); err != nil {
		return "", err
	}

	return tmpFile, nil
}

func executeScript(file string) error {
	cwd, err := os.Getwd()
	if err != nil {
		return fmt.Errorf("failed to get current directory: %v", err)
	}

	cmd := exec.Command("bash", file)
	cmd.Dir = cwd
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
}

func executeDirectCommand(command string) error {
	cwd, err := os.Getwd()
	if err != nil {
		return fmt.Errorf("failed to get current directory: %v", err)
	}

	cmd := exec.Command("bash", "-c", command)
	cmd.Dir = cwd
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
}
