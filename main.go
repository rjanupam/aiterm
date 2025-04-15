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
	"github.com/rjanupam/aiterm/config"
)

func main() {
	ctx := context.Background()

	cfg, err := config.LoadConfig()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to load configuration: %v\n", err)
		os.Exit(1)
	}

	modelName := os.Getenv("AITERM_MODEL")
	if modelName == "" {
		modelName = cfg.Model
	}

	aiClient, err := ai.NewAIClient(ctx, modelName, cfg)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to initialize AI client: %v\n", err)
		os.Exit(1)
	}
	defer aiClient.Close()

	reader := bufio.NewReader(os.Stdin)

	currentDir, err := os.Getwd()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error getting current directory: %v\n", err)
		currentDir = "."
	}

	for {
		shortDir := filepath.Base(currentDir)
		if shortDir == "." {
			shortDir = filepath.Base(filepath.Dir(currentDir))
		}
		fmt.Printf("aiterm:%s> ", shortDir)

		input, err := reader.ReadString('\n')
		if err != nil {
			if err.Error() != "EOF" {
				fmt.Println("\nExiting.")
				break
			}
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

		if input == "config" {
			configPath, err := config.GetConfigPath()
			if err != nil {
				fmt.Fprintf(os.Stderr, "Error getting config path: %v\n", err)
				continue
			}
			fmt.Printf("Opening config file: %s\n", configPath)
			if err := openConfigFile(configPath); err != nil {
				fmt.Fprintf(os.Stderr, "Error opening config file: %v\n", err)
			}
			continue
		}

		if strings.HasPrefix(input, "$") {
			command := strings.TrimSpace(input[1:])
			if command == "" {
				continue
			}

			if strings.HasPrefix(command, "cd ") {
				newDir := strings.TrimSpace(command[3:])
				if newDir == "" {
					homeDir, err := os.UserHomeDir()
					if err != nil {
						fmt.Fprintf(os.Stderr, "Error getting home directory: %v\n", err)
						continue
					}
					newDir = homeDir
				} else if !filepath.IsAbs(newDir) {
					newDir = filepath.Join(currentDir, newDir)
				}

				if _, err := os.Stat(newDir); os.IsNotExist(err) {
					fmt.Fprintf(os.Stderr, "Directory does not exist: %s\n", newDir)
					continue
				}

				currentDir = newDir
				continue
			}

			if err := executeDirectCommand(command, currentDir); err != nil {
				fmt.Fprintf(os.Stderr, "Error executing command: %v\n", err)
			}
			continue
		}

		fmt.Print("AI: ")
		response, err := aiClient.ProcessPrompt(ctx, input, os.Stdout)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error processing prompt: %v\n", err)
			continue
		}

		codeBlock := extractBashCodeBlock(response)
		if codeBlock != "" {
			var newDir string
			var cdTarget string
			if strings.Contains(codeBlock, "cd ") {
				cdRegex := regexp.MustCompile(`cd\s+([^\s;]+)`)
				matches := cdRegex.FindStringSubmatch(codeBlock)
				if len(matches) > 1 {
					cdTarget = matches[1]
					newDir = cdTarget

					if newDir == ".." {
						newDir = filepath.Dir(currentDir)
					} else if newDir == "." {
						newDir = currentDir
					} else if newDir == "~" {
						homeDir, err := os.UserHomeDir()
						if err != nil {
							fmt.Fprintf(os.Stderr, "Error getting home directory: %v\n", err)
							newDir = currentDir
						} else {
							newDir = homeDir
						}
					} else if !filepath.IsAbs(newDir) {
						newDir = filepath.Join(currentDir, newDir)
					}

					if _, err := os.Stat(newDir); os.IsNotExist(err) {
						fmt.Fprintf(os.Stderr, "Directory does not exist: %s\n", newDir)
						newDir = currentDir
					}
				}
			}

			if newDir != "" && newDir != currentDir {
				currentDir = newDir
				fmt.Printf("Changed directory to: %s\n", currentDir)

				if cdTarget != "" && strings.TrimSpace(codeBlock) == "cd "+cdTarget {
					continue
				}
			}

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
			fmt.Println("-------------------- Proposed Script --------------------")
			fmt.Print(string(content))
			fmt.Println("---------------------------------------------------------")

			fmt.Print("Execute this script? (Y/n): ")
			confirm, _ := reader.ReadString('\n')
			confirm = strings.TrimSpace(strings.ToLower(confirm))

			if confirm == "" || confirm == "y" {
				if err := executeScript(tmpFile, currentDir); err != nil {
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

	if _, err := f.WriteString("#!/bin/bash\n"); err != nil {
		f.Close()
		os.Remove(tmpFile)
		return "", fmt.Errorf("failed to write shebang: %v", err)
	}
	if _, err := f.WriteString(code); err != nil {
		f.Close()
		os.Remove(tmpFile)
		return "", fmt.Errorf("failed to write code: %w", err)
	}

	if !strings.HasSuffix(code, "\n") {
		if _, err := f.WriteString("\n"); err != nil {
			f.Close()
			os.Remove(tmpFile)
			return "", fmt.Errorf("failed to write trailing newline: %w", err)
		}
	}

	if err := f.Close(); err != nil {
		os.Remove(tmpFile)
		return "", fmt.Errorf("failed to close temp file: %w", err)
	}

	if err := os.Chmod(tmpFile, 0700); err != nil {
		os.Remove(tmpFile)
		return "", fmt.Errorf("failed to set executable permission: %w", err)
	}

	return tmpFile, nil
}

func executeScript(file string, workingDir string) error {
	cmd := exec.Command("bash", file)
	cmd.Dir = workingDir
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
}

func executeDirectCommand(command string, workingDir string) error {
	cmd := exec.Command("bash", "-c", command)
	cmd.Dir = workingDir
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
}

func openConfigFile(configPath string) error {
	editor := os.Getenv("EDITOR")
	if editor == "" {
		editor = "vim"
	}

	cmd := exec.Command(editor, configPath)
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
}
