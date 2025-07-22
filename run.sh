#!/bin/bash
# Document Similarity Tool - Unix/Linux/macOS Shell Script
# This script provides easy access to the document similarity system

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_color() {
    printf "${!1}%s${NC}\n" "$2"
}

# Check if Python is available
check_python() {
    if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
        print_color RED "ERROR: Python is not installed or not in PATH"
        print_color RED "Please install Python 3.8+ and try again"
        exit 1
    fi
    
    # Use python3 if available, otherwise python
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    else
        PYTHON_CMD="python"
    fi
}

# Main menu
show_menu() {
    echo
    print_color BLUE "========================================"
    print_color BLUE "Document Similarity Detection System"
    print_color BLUE "========================================"
    echo
    echo "Please choose an option:"
    echo
    echo "1. Setup and verify system"
    echo "2. Run simple example"
    echo "3. Run full demo"
    echo "4. Process documents in sample_pdfs"
    echo "5. Search for similar documents"
    echo "6. Find duplicates"
    echo "7. Show help"
    echo "8. Exit"
    echo
}

# Setup function
run_setup() {
    echo
    print_color YELLOW "Running setup and verification..."
    $PYTHON_CMD setup.py
    read -p "Press Enter to continue..."
}

# Simple example function
run_example() {
    echo
    print_color YELLOW "Running simple example..."
    $PYTHON_CMD example.py
    read -p "Press Enter to continue..."
}

# Demo function
run_demo() {
    echo
    print_color YELLOW "Running full demonstration..."
    $PYTHON_CMD main.py demo
    read -p "Press Enter to continue..."
}

# Process documents function
run_process() {
    echo
    print_color YELLOW "Processing documents in data/sample_pdfs..."
    $PYTHON_CMD main.py process -d data/sample_pdfs
    read -p "Press Enter to continue..."
}

# Search function
run_search() {
    echo
    read -p "Enter search query: " query
    echo
    print_color YELLOW "Searching for: $query"
    $PYTHON_CMD main.py search -q "$query" -k 5
    read -p "Press Enter to continue..."
}

# Find duplicates function
run_duplicates() {
    echo
    print_color YELLOW "Finding duplicate documents..."
    $PYTHON_CMD main.py duplicates -t 0.9
    read -p "Press Enter to continue..."
}

# Show help function
show_help() {
    echo
    print_color GREEN "Available commands:"
    echo "$PYTHON_CMD setup.py                           # Setup and verify system"
    echo "$PYTHON_CMD example.py                         # Run simple example"
    echo "$PYTHON_CMD main.py demo                       # Run full demo"
    echo "$PYTHON_CMD main.py process -d data/sample_pdfs  # Process directory"
    echo "$PYTHON_CMD main.py search -q 'query'          # Search documents"
    echo "$PYTHON_CMD main.py cluster -t 0.7             # Cluster documents"
    echo "$PYTHON_CMD main.py duplicates                 # Find duplicates"
    echo "$PYTHON_CMD main.py --help                     # Show detailed help"
    echo
    print_color GREEN "For detailed documentation, see README.md"
    read -p "Press Enter to continue..."
}

# Main script
main() {
    # Check Python availability
    check_python
    
    while true; do
        show_menu
        read -p "Enter your choice (1-8): " choice
        
        case $choice in
            1)
                run_setup
                ;;
            2)
                run_example
                ;;
            3)
                run_demo
                ;;
            4)
                run_process
                ;;
            5)
                run_search
                ;;
            6)
                run_duplicates
                ;;
            7)
                show_help
                ;;
            8)
                echo
                print_color GREEN "Thank you for using Document Similarity Detection System!"
                exit 0
                ;;
            *)
                print_color RED "Invalid choice. Please try again."
                sleep 1
                ;;
        esac
    done
}

# Run main function
main "$@"
