# Test runner script for Medical RAG Chatbot (Windows PowerShell)

Write-Host "🏥 Medical RAG Chatbot - Test Runner" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan

$args = $args -join " "

switch -Regex ($args) {
    "(--all|-a|all)" {
        Write-Host "Running all tests..." -ForegroundColor Yellow
        python run_tests.py --all
    }
    "(--unit|-u|unit)" {
        Write-Host "Running unit tests..." -ForegroundColor Yellow
        python run_tests.py --unit
    }
    "(--integration|-i|integration)" {
        Write-Host "Running integration tests..." -ForegroundColor Yellow
        python run_tests.py --integration
    }
    "(--security|-s|security)" {
        Write-Host "Running security tests..." -ForegroundColor Yellow
        python run_tests.py --security
    }
    "(--fast|-f|fast)" {
        Write-Host "Running fast tests (excluding slow)..." -ForegroundColor Yellow
        python run_tests.py --fast
    }
    "(--coverage|-c|coverage)" {
        Write-Host "Running all tests with coverage..." -ForegroundColor Yellow
        python run_tests.py --cov
    }
    "(--help|-h|help)" {
        Write-Host @"
Usage: .\run_tests.ps1 [option]

Options:
  --all, -a        Run all tests
  --unit, -u       Run unit tests only
  --integration, -i Run integration tests only
  --security, -s   Run security tests only
  --fast, -f       Run fast tests (exclude slow)
  --coverage, -c   Run all tests with coverage
  --help, -h       Show this help message

Examples:
  .\run_tests.ps1 --unit
  .\run_tests.ps1 --fast
  .\run_tests.ps1 --coverage
"@
    }
    default {
        Write-Host "Running all tests (excluding slow)..." -ForegroundColor Yellow
        python run_tests.py --fast
    }
}