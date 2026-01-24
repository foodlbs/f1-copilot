'use client';

import { useState, useRef, useEffect } from 'react';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  sources?: any[];
  strategy?: string;
}

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [strategy, setStrategy] = useState('similarity');
  const [streaming, setStreaming] = useState(true);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const sendMessage = async () => {
    if (!input.trim()) return;

    const userMessage: Message = { role: 'user', content: input };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setLoading(true);

    if (streaming) {
      await sendStreamingMessage(input);
    } else {
      await sendRegularMessage(input);
    }
  };

  const sendStreamingMessage = async (query: string) => {
    try {
      const response = await fetch('http://localhost:8000/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query,
          session_id: sessionId,
          retrieval_strategy: strategy,
          stream: true,
          top_k: 5
        }),
      });

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();
      let assistantMessage = '';

      setMessages(prev => [...prev, {
        role: 'assistant',
        content: '',
        strategy: strategy
      }]);

      if (reader) {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          const chunk = decoder.decode(value);
          const lines = chunk.split('\n');

          for (const line of lines) {
            if (line.startsWith('data: ')) {
              const data = JSON.parse(line.slice(6));

              if (data.type === 'token') {
                assistantMessage += data.token;
                setMessages(prev => {
                  const newMessages = [...prev];
                  newMessages[newMessages.length - 1].content = assistantMessage;
                  return newMessages;
                });
              } else if (data.type === 'sources') {
                setMessages(prev => {
                  const newMessages = [...prev];
                  newMessages[newMessages.length - 1].sources = data.sources;
                  newMessages[newMessages.length - 1].strategy = data.strategy;
                  return newMessages;
                });
              } else if (data.type === 'done') {
                setLoading(false);
              } else if (data.type === 'error') {
                console.error('Stream error:', data.error);
                setLoading(false);
              }
            }
          }
        }
      }
    } catch (error) {
      console.error('Error:', error);
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: 'Sorry, an error occurred.',
      }]);
      setLoading(false);
    }
  };

  const sendRegularMessage = async (query: string) => {
    try {
      const response = await fetch('http://localhost:8000/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query,
          session_id: sessionId,
          retrieval_strategy: strategy,
          stream: false,
          top_k: 5
        }),
      });

      const data = await response.json();

      if (!sessionId) {
        setSessionId(data.session_id);
      }

      setMessages(prev => [...prev, {
        role: 'assistant',
        content: data.answer,
        sources: data.sources,
        strategy: data.strategy_used
      }]);
    } catch (error) {
      console.error('Error:', error);
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: 'Sorry, an error occurred.',
      }]);
    } finally {
      setLoading(false);
    }
  };

  const clearChat = async () => {
    if (sessionId) {
      await fetch(`http://localhost:8000/api/session/${sessionId}`, {
        method: 'DELETE',
      });
    }
    setMessages([]);
    setSessionId(null);
  };

  const testStrategies = async () => {
    if (!input.trim()) return;

    setLoading(true);
    try {
      const response = await fetch(
        `http://localhost:8000/api/test-strategies?query=${encodeURIComponent(input)}&top_k=5`,
        { method: 'POST' }
      );
      const data = await response.json();

      alert(`Strategy Comparison:\n\n${JSON.stringify(data, null, 2)}`);
    } catch (error) {
      console.error('Error:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-screen max-w-7xl mx-auto p-4">
      {/* Header */}
      <div className="flex justify-between items-center mb-4 p-4 bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg shadow-lg">
        <h1 className="text-3xl font-bold text-white">
          Advanced RAG Chatbot
        </h1>
        <div className="flex gap-2">
          <button
            onClick={testStrategies}
            disabled={!input.trim()}
            className="px-4 py-2 bg-yellow-500 text-white rounded hover:bg-yellow-600 disabled:opacity-50"
          >
            Test Strategies
          </button>
          <button
            onClick={clearChat}
            className="px-4 py-2 bg-red-500 text-white rounded hover:bg-red-600"
          >
            Clear Chat
          </button>
        </div>
      </div>

      {/* Settings Panel */}
      <div className="mb-4 p-4 bg-white rounded-lg shadow">
        <div className="flex gap-4 items-center">
          <label className="flex items-center gap-2">
            <span className="font-semibold">Retrieval Strategy:</span>
            <select
              value={strategy}
              onChange={(e) => setStrategy(e.target.value)}
              className="px-3 py-2 border rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="similarity">Similarity (Fast)</option>
              <option value="mmr">MMR (Diverse)</option>
              <option value="multi_query">Multi-Query (Comprehensive)</option>
              <option value="compression">Compression (Focused)</option>
            </select>
          </label>

          <label className="flex items-center gap-2">
            <input
              type="checkbox"
              checked={streaming}
              onChange={(e) => setStreaming(e.target.checked)}
              className="w-4 h-4"
            />
            <span className="font-semibold">Enable Streaming</span>
          </label>
        </div>

        <div className="mt-2 text-sm text-gray-600">
          {strategy === 'similarity' && 'Best for straightforward questions'}
          {strategy === 'mmr' && 'Best for diverse perspectives'}
          {strategy === 'multi_query' && 'Best for complex questions (slower)'}
          {strategy === 'compression' && 'Best for extracting specific info (slower)'}
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto space-y-4 mb-4 bg-gray-50 p-4 rounded-lg">
        {messages.map((msg, idx) => (
          <div
            key={idx}
            className={`p-4 rounded-lg ${
              msg.role === 'user'
                ? 'bg-blue-100 ml-auto max-w-[70%]'
                : 'bg-white mr-auto max-w-[80%] shadow-md'
            }`}
          >
            <p className="whitespace-pre-wrap">{msg.content}</p>

            {msg.strategy && (
              <div className="mt-2 text-xs text-gray-500">
                Strategy: {msg.strategy}
              </div>
            )}

            {msg.sources && msg.sources.length > 0 && (
              <details className="mt-3 text-sm">
                <summary className="cursor-pointer font-semibold text-gray-700 hover:text-blue-600">
                  Sources ({msg.sources.length})
                </summary>
                <div className="mt-2 space-y-2 max-h-60 overflow-y-auto">
                  {msg.sources.map((src, i) => (
                    <div key={i} className="p-3 bg-gray-50 rounded border border-gray-200">
                      <div className="flex justify-between items-start mb-1">
                        <span className="text-xs font-semibold text-blue-600">
                          Source {i + 1}
                        </span>
                        {src.metadata?.filename && (
                          <span className="text-xs text-gray-500">
                            {src.metadata.filename}
                          </span>
                        )}
                      </div>
                      <p className="text-sm text-gray-700">{src.content}</p>
                    </div>
                  ))}
                </div>
              </details>
            )}
          </div>
        ))}

        {loading && (
          <div className="bg-white p-4 rounded-lg shadow-md mr-auto max-w-[70%]">
            <div className="flex items-center space-x-2">
              <div className="flex space-x-1">
                <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce"></div>
                <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce delay-100"></div>
                <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce delay-200"></div>
              </div>
              <span className="text-sm text-gray-500">
                {streaming ? 'Streaming response...' : 'Thinking...'}
              </span>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="flex gap-2">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault();
              sendMessage();
            }
          }}
          placeholder="Ask anything... (Try different retrieval strategies)"
          className="flex-1 p-4 border-2 border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          disabled={loading}
        />
        <button
          onClick={sendMessage}
          disabled={loading || !input.trim()}
          className="px-8 py-4 bg-gradient-to-r from-blue-500 to-purple-600 text-white rounded-lg hover:from-blue-600 hover:to-purple-700 disabled:opacity-50 disabled:cursor-not-allowed font-semibold shadow-lg transform transition hover:scale-105"
        >
          {loading ? 'Sending...' : 'Send'}
        </button>
      </div>

      {sessionId && (
        <div className="mt-2 text-xs text-gray-500 text-center">
          Session ID: {sessionId.slice(0, 8)}...
        </div>
      )}
    </div>
  );
}
