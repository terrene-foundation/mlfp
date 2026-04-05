// Copyright 2026 Terrene Foundation
// SPDX-License-Identifier: Apache-2.0
//! Kailash SDK Textbook -- Agents / Streaming Chat
//!
//! OBJECTIVE: Build a streaming chat agent with conversation history.
//! LEVEL: Intermediate
//! PARITY: Equivalent -- Python uses async generators for streaming;
//!         Rust uses tokio channels or async Stream trait.
//! VALIDATES: Streaming patterns, conversation history, message types
//!
//! Run: cargo run -p tutorial-agents --bin 06_streaming_chat

use kaizen_agents::delegate_engine::DelegateEvent;
use kaizen_agents::history::{ChatMessage, ConversationHistory, Role};

fn main() {
    // ── 1. Conversation history ──
    // ChatMessage represents a single message in a conversation.
    // ConversationHistory maintains the full thread.

    let mut history = ConversationHistory::new();
    assert_eq!(history.len(), 0);

    // ── 2. Message roles ──
    // Messages have roles: System, User, Assistant, Tool

    assert_eq!(Role::System.as_str(), "system");
    assert_eq!(Role::User.as_str(), "user");
    assert_eq!(Role::Assistant.as_str(), "assistant");
    assert_eq!(Role::Tool.as_str(), "tool");

    // ── 3. Add messages to history ──

    history.add(ChatMessage::system(
        "You are a helpful assistant specializing in Rust programming."
    ));
    history.add(ChatMessage::user("What is ownership in Rust?"));
    history.add(ChatMessage::assistant(
        "Ownership is Rust's memory management system. Each value has \
         exactly one owner, and the value is dropped when the owner \
         goes out of scope."
    ));

    assert_eq!(history.len(), 3);

    // ── 4. Message inspection ──

    let first = &history.messages()[0];
    assert_eq!(first.role(), Role::System);
    assert!(first.content().contains("Rust programming"));

    let last = history.last().unwrap();
    assert_eq!(last.role(), Role::Assistant);

    // ── 5. History windowing ──
    // For long conversations, limit context to recent messages.
    // This prevents exceeding the model's context window.

    history.add(ChatMessage::user("How does borrowing work?"));
    history.add(ChatMessage::assistant(
        "Borrowing lets you reference data without taking ownership."
    ));

    let recent = history.last_n(3);
    assert_eq!(recent.len(), 3);

    // ── 6. Streaming event consumption ──
    // In a streaming chat, the agent yields DelegateEvent variants:
    //
    //   TextDelta { text }  -- partial text tokens
    //   TurnComplete { .. } -- full response ready
    //
    // The streaming pattern in Rust:
    //
    //   let mut full_text = String::new();
    //   while let Some(event) = stream.next().await {
    //       match event {
    //           DelegateEvent::TextDelta { text } => {
    //               print!("{text}");
    //               full_text.push_str(&text);
    //           }
    //           DelegateEvent::TurnComplete { .. } => {
    //               println!();
    //               history.add(ChatMessage::assistant(&full_text));
    //               break;
    //           }
    //           _ => {}
    //       }
    //   }

    // Simulate streaming events
    let events = vec![
        DelegateEvent::TextDelta { text: "Borrow".to_string() },
        DelegateEvent::TextDelta { text: "ing allows".to_string() },
        DelegateEvent::TextDelta { text: " references.".to_string() },
        DelegateEvent::TurnComplete {
            text: "Borrowing allows references.".to_string(),
            prompt_tokens: 150,
            completion_tokens: 10,
        },
    ];

    let mut streamed = String::new();
    for event in &events {
        match event {
            DelegateEvent::TextDelta { text } => {
                streamed.push_str(text);
            }
            DelegateEvent::TurnComplete { text, .. } => {
                assert_eq!(&streamed, text);
            }
            _ => {}
        }
    }
    assert_eq!(streamed, "Borrowing allows references.");

    // ── 7. History serialization ──
    // ConversationHistory serializes to JSON for persistence or API calls.

    let json = history.to_json();
    assert!(json.contains("ownership"));
    assert!(json.contains("borrowing"));

    // ── 8. Key concepts ──
    // - ConversationHistory: maintains chat thread
    // - ChatMessage: role + content for each message
    // - Role: System, User, Assistant, Tool
    // - Streaming: TextDelta events for real-time display
    // - TurnComplete: signals full response is ready
    // - History windowing: last_n() for context management
    // - Python uses async generators; Rust uses async Stream

    println!("PASS: 04-agents/06_streaming_chat");
}
