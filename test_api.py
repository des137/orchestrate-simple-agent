#!/usr/bin/env python3
"""
Simple test script for the FastAPI agent.
Tests the chat completions endpoint with various queries.
"""

import requests


def test_endpoint(base_url: str, test_name: str, user_message: str):
    """Test the chat completions endpoint with a user message."""
    print(f"\n{'=' * 60}")
    print(f"Test: {test_name}")
    print(f"{'=' * 60}")
    print(f"User: {user_message}")

    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": user_message}],
        "stream": False,
    }

    try:
        response = requests.post(
            f"{base_url}/chat/completions", json=payload, timeout=30
        )

        if response.status_code == 200:
            data = response.json()
            assistant_message = data["choices"][0]["message"]["content"]
            print(f"Assistant: {assistant_message}")
            print("✅ Success")
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")

    except Exception as e:
        print(f"❌ Exception: {str(e)}")


def main():
    """Run all tests."""
    base_url = "http://localhost:8080"

    # Test health endpoint first
    print("Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            print("✅ Server is healthy")
        else:
            print(f"❌ Server returned status {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        print(f"Make sure the server is running on {base_url}")
        return

    # Run chat tests
    print("\n" + "=" * 60)
    print("Running Chat Completion Tests")
    print("=" * 60)

    test_endpoint(
        base_url, "Test 1: Simple Greeting", "Hi, my name is Alice! Nice to meet you."
    )

    test_endpoint(base_url, "Test 2: Math - Addition", "What is 25 plus 17?")

    test_endpoint(base_url, "Test 3: Math - Multiplication", "Calculate 8 times 9")

    test_endpoint(
        base_url,
        "Test 4: Math - Division",
        "Divide 144 by 12",
    )

    test_endpoint(
        base_url,
        "Test 5: Complex Math",
        "Add 10 and 5, then multiply the result by 3",
    )

    test_endpoint(base_url, "Test 6: General Question", "What can you help me with?")

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
