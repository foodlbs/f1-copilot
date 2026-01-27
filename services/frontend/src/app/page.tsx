'use client';

import { useState, useRef, useEffect } from 'react';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  sources?: any[];
  strategy?: string;
  webSearchUsed?: boolean;
  cachedForFuture?: boolean;
}

interface StrategyResult {
  query: string;
  results: Record<string, any>;
}

// F1 Flag Icon Component
const FlagIcon = () => (
  <svg className="w-6 h-6" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
    <path d="M4 21V4" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
    <path d="M4 4H20L17 8.5L20 13H4" fill="#e10600" stroke="#e10600" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
  </svg>
);

// Racing Helmet Icon
const HelmetIcon = () => (
  <svg className="w-5 h-5" viewBox="0 0 24 24" fill="currentColor">
    <path d="M12 2C6.48 2 2 6.48 2 12c0 3.69 2.47 6.86 6 8.25V22h8v-1.75c3.53-1.39 6-4.56 6-8.25 0-5.52-4.48-10-10-10zm0 2c4.41 0 8 3.59 8 8 0 2.21-.89 4.21-2.34 5.66l-1.41-1.41C17.36 15.14 18 13.64 18 12c0-3.31-2.69-6-6-6s-6 2.69-6 6c0 1.64.64 3.14 1.75 4.25l-1.41 1.41C4.89 16.21 4 14.21 4 12c0-4.41 3.59-8 8-8z"/>
  </svg>
);

// Steering Wheel Icon
const SteeringIcon = () => (
  <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <circle cx="12" cy="12" r="9"/>
    <circle cx="12" cy="12" r="3"/>
    <path d="M12 3v6M3 12h6M21 12h-6"/>
  </svg>
);

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [strategy, setStrategy] = useState('similarity');
  const [streaming, setStreaming] = useState(true);
  const [showStrategyModal, setShowStrategyModal] = useState(false);
  const [strategyResults, setStrategyResults] = useState<StrategyResult | null>(null);
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
        let buffer = '';

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n');
          // Keep the last incomplete line in buffer
          buffer = lines.pop() || '';

          for (const line of lines) {
            if (line.startsWith('data: ')) {
              try {
                const data = JSON.parse(line.slice(6));

                if (data.type === 'token') {
                  assistantMessage += data.token;
                  setMessages(prev => {
                    const newMessages = [...prev];
                    newMessages[newMessages.length - 1].content = assistantMessage;
                    return newMessages;
                  });
                } else if (data.type === 'web_search_start') {
                  // Clear current content and show searching indicator
                  assistantMessage = '';
                  setMessages(prev => {
                    const newMessages = [...prev];
                    newMessages[newMessages.length - 1].content = '🔍 Searching the web for more information...';
                    newMessages[newMessages.length - 1].webSearchUsed = true;
                    return newMessages;
                  });
                } else if (data.type === 'web_answer_start') {
                  // Clear the searching message before new answer streams in
                  assistantMessage = '';
                  setMessages(prev => {
                    const newMessages = [...prev];
                    newMessages[newMessages.length - 1].content = '';
                    return newMessages;
                  });
                } else if (data.type === 'sources') {
                  setMessages(prev => {
                    const newMessages = [...prev];
                    newMessages[newMessages.length - 1].sources = data.sources;
                    newMessages[newMessages.length - 1].strategy = data.strategy;
                    newMessages[newMessages.length - 1].webSearchUsed = data.web_search_used || false;
                    newMessages[newMessages.length - 1].cachedForFuture = data.cached_for_future || false;
                    return newMessages;
                  });
                } else if (data.type === 'done') {
                  setLoading(false);
                } else if (data.type === 'error') {
                  console.error('Stream error:', data.error);
                  setLoading(false);
                }
              } catch (parseError) {
                // Skip malformed JSON lines
                console.warn('Skipping malformed SSE line:', line);
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
        strategy: data.strategy_used,
        webSearchUsed: data.web_search_used || false,
        cachedForFuture: data.cached_for_future || false
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
      setStrategyResults(data);
      setShowStrategyModal(true);
    } catch (error) {
      console.error('Error:', error);
    } finally {
      setLoading(false);
    }
  };

  const sendSuggestion = async (suggestion: string) => {
    setInput(suggestion);
    const userMessage: Message = { role: 'user', content: suggestion };
    setMessages(prev => [...prev, userMessage]);
    setLoading(true);

    if (streaming) {
      await sendStreamingMessage(suggestion);
    } else {
      await sendRegularMessage(suggestion);
    }
  };

  const strategyDescriptions: Record<string, string> = {
    similarity: 'Fastest lap time - Quick similarity search',
    mmr: 'Diverse grid positions - MMR for varied results',
    multi_query: 'Full race analysis - Multi-query for depth',
    compression: 'Pit stop precision - Compressed relevant info'
  };

  return (
    <div className="flex flex-col h-screen max-w-7xl mx-auto p-4">
      {/* Header */}
      <div className="f1-card mb-4 p-4 glow-red">
        <div className="flex justify-between items-center">
          <div className="flex items-center gap-4">
            <div className="w-12 h-12 rounded-full bg-gradient-to-br from-[#e10600] to-[#b30500] flex items-center justify-center glow-red">
              <FlagIcon />
            </div>
            <div>
              <h1 className="text-3xl font-bold text-white tracking-tight">
                F1 <span className="text-gradient-f1">Strategy</span> Assistant
              </h1>
              <p className="text-gray-400 text-sm">Powered by AI Race Intelligence</p>
            </div>
          </div>
          <div className="flex gap-3">
            <button
              onClick={testStrategies}
              disabled={!input.trim()}
              className="f1-button-secondary flex items-center gap-2"
            >
              <SteeringIcon />
              Compare Strategies
            </button>
            <button
              onClick={clearChat}
              className="f1-button-secondary hover:bg-red-900/50 hover:border-red-500"
            >
              Clear Session
            </button>
          </div>
        </div>
      </div>

      {/* Settings Panel */}
      <div className="f1-card mb-4 p-4">
        <div className="flex flex-wrap gap-6 items-center">
          <div className="flex items-center gap-3">
            <span className="text-gray-400 font-semibold text-sm uppercase tracking-wider">Strategy Mode:</span>
            <select
              value={strategy}
              onChange={(e) => setStrategy(e.target.value)}
              className="f1-select"
            >
              <option value="similarity">Qualifying (Fast)</option>
              <option value="mmr">Race (Diverse)</option>
              <option value="multi_query">Full Analysis (Deep)</option>
              <option value="compression">Pit Strategy (Focused)</option>
            </select>
          </div>

          <label className="flex items-center gap-2 cursor-pointer">
            <div className={`relative w-12 h-6 rounded-full transition-colors ${streaming ? 'bg-[#e10600]' : 'bg-[#38383f]'}`}>
              <input
                type="checkbox"
                checked={streaming}
                onChange={(e) => setStreaming(e.target.checked)}
                className="sr-only"
              />
              <div className={`absolute top-1 w-4 h-4 rounded-full bg-white transition-transform ${streaming ? 'translate-x-7' : 'translate-x-1'}`}></div>
            </div>
            <span className="text-gray-300 font-medium">Live Telemetry</span>
          </label>

          <div className="flex-1 text-right">
            <span className="text-xs text-gray-500 italic">
              {strategyDescriptions[strategy]}
            </span>
          </div>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto space-y-4 mb-4 f1-card p-4">
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full text-center">
            <div className="w-24 h-24 mb-6 rounded-full bg-gradient-to-br from-[#e10600]/20 to-[#b30500]/20 flex items-center justify-center">
              <svg className="w-12 h-12 text-[#e10600]" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/>
              </svg>
            </div>
            <h2 className="text-2xl font-bold text-white mb-2">Welcome to the Pit Wall</h2>
            <p className="text-gray-400 max-w-md">
              Ask me anything about F1 race strategies, driver performance, pit stops, and historical data.
              I'm your AI race engineer ready to analyze the grid.
            </p>
            <div className="mt-6 flex flex-wrap gap-2 justify-center">
              {[
                "Best pit stop strategies for Monaco?",
                "Compare HAM vs VER 2021 season",
                "Optimal tire strategy for hot weather",
                "Most successful team strategies"
              ].map((suggestion, i) => (
                <button
                  key={i}
                  onClick={() => sendSuggestion(suggestion)}
                  className="px-4 py-2 text-sm bg-[#38383f]/50 text-gray-300 rounded-full hover:bg-[#e10600]/20 hover:text-[#e10600] transition-colors border border-[#38383f] hover:border-[#e10600]/50"
                >
                  {suggestion}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((msg, idx) => (
          <div
            key={idx}
            className={`p-4 rounded-lg ${
              msg.role === 'user'
                ? 'bg-gradient-to-r from-[#e10600]/20 to-[#b30500]/10 border border-[#e10600]/30 ml-auto max-w-[70%]'
                : 'bg-[#1f1f27] border border-[#38383f] mr-auto max-w-[80%] racing-stripe pl-6'
            }`}
          >
            <div className="flex items-start gap-3">
              {msg.role === 'assistant' && (
                <div className="w-8 h-8 rounded-full bg-gradient-to-br from-[#e10600] to-[#b30500] flex items-center justify-center flex-shrink-0">
                  <HelmetIcon />
                </div>
              )}
              <div className="flex-1">
                <p className="text-gray-100 whitespace-pre-wrap">{msg.content}</p>

                <div className="mt-2 flex flex-wrap gap-2">
                  {msg.strategy && (
                    <div className="inline-flex items-center gap-1 text-xs text-[#e10600] bg-[#e10600]/10 px-2 py-1 rounded">
                      <SteeringIcon />
                      {msg.strategy}
                    </div>
                  )}
                  {msg.webSearchUsed && (
                    <div className="inline-flex items-center gap-1 text-xs text-blue-400 bg-blue-400/10 px-2 py-1 rounded">
                      <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M21 12a9 9 0 01-9 9m9-9a9 9 0 00-9-9m9 9H3m9 9a9 9 0 01-9-9m9 9c1.657 0 3-4.03 3-9s-1.343-9-3-9m0 18c-1.657 0-3-4.03-3-9s1.343-9 3-9m-9 9a9 9 0 019-9"/>
                      </svg>
                      Web Search
                    </div>
                  )}
                  {msg.cachedForFuture && (
                    <div className="inline-flex items-center gap-1 text-xs text-green-400 bg-green-400/10 px-2 py-1 rounded">
                      <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7"/>
                      </svg>
                      Cached
                    </div>
                  )}
                </div>

                {msg.sources && msg.sources.length > 0 && (
                  <details className="mt-4 text-sm">
                    <summary className="cursor-pointer font-semibold text-gray-400 hover:text-[#e10600] transition-colors flex items-center gap-2">
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/>
                      </svg>
                      Race Data Sources ({msg.sources.length})
                    </summary>
                    <div className="mt-3 space-y-2 max-h-60 overflow-y-auto">
                      {msg.sources.map((src, i) => (
                        <div key={i} className="p-3 bg-[#15151e] rounded border border-[#38383f]">
                          <div className="flex justify-between items-start mb-2">
                            <span className="text-xs font-bold text-[#e10600] uppercase tracking-wider">
                              Lap {i + 1} Data
                            </span>
                            {src.metadata?.driver && (
                              <span className="text-xs bg-[#38383f] px-2 py-0.5 rounded text-gray-300">
                                {src.metadata.driver}
                              </span>
                            )}
                            {src.metadata?.season && (
                              <span className="text-xs text-gray-500">
                                Season {src.metadata.season}
                              </span>
                            )}
                          </div>
                          <p className="text-sm text-gray-400">{src.content}</p>
                        </div>
                      ))}
                    </div>
                  </details>
                )}
              </div>
              {msg.role === 'user' && (
                <div className="w-8 h-8 rounded-full bg-[#38383f] flex items-center justify-center flex-shrink-0">
                  <svg className="w-4 h-4 text-gray-300" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M10 9a3 3 0 100-6 3 3 0 000 6zm-7 9a7 7 0 1114 0H3z" clipRule="evenodd"/>
                  </svg>
                </div>
              )}
            </div>
          </div>
        ))}

        {loading && (
          <div className="bg-[#1f1f27] border border-[#38383f] p-4 rounded-lg mr-auto max-w-[70%] racing-stripe pl-6">
            <div className="flex items-center space-x-3">
              <div className="w-8 h-8 rounded-full bg-gradient-to-br from-[#e10600] to-[#b30500] flex items-center justify-center animate-pulse">
                <HelmetIcon />
              </div>
              <div className="flex-1">
                <div className="h-2 w-full bg-[#38383f] rounded-full overflow-hidden">
                  <div className="h-full w-8 bg-gradient-to-r from-[#e10600] to-[#ff6b6b] rounded-full race-loader"></div>
                </div>
                <span className="text-xs text-gray-500 mt-2 block">
                  {streaming ? 'Streaming live telemetry...' : 'Analyzing race data...'}
                </span>
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="f1-card p-4">
        <div className="flex gap-3">
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
            placeholder="Ask your race engineer... (e.g., 'What's the best strategy for Monza?')"
            className="flex-1 p-4 f1-input text-lg"
            disabled={loading}
          />
          <button
            onClick={sendMessage}
            disabled={loading || !input.trim()}
            className="f1-button-primary px-8 flex items-center gap-2"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 10V3L4 14h7v7l9-11h-7z"/>
            </svg>
            {loading ? 'Racing...' : 'Send'}
          </button>
        </div>
      </div>

      {sessionId && (
        <div className="mt-3 text-xs text-gray-600 text-center flex items-center justify-center gap-2">
          <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></div>
          Session Active: {sessionId.slice(0, 8)}...
        </div>
      )}

      {/* Strategy Comparison Modal */}
      {showStrategyModal && strategyResults && (
        <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50 p-4">
          <div className="f1-card max-w-4xl w-full max-h-[80vh] overflow-hidden flex flex-col">
            <div className="p-4 border-b border-[#38383f] flex justify-between items-center">
              <div>
                <h2 className="text-xl font-bold text-white">Strategy Comparison</h2>
                <p className="text-sm text-gray-400 mt-1">Query: "{strategyResults.query}"</p>
              </div>
              <button
                onClick={() => setShowStrategyModal(false)}
                className="w-8 h-8 rounded-full bg-[#38383f] hover:bg-[#e10600] transition-colors flex items-center justify-center"
              >
                <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12"/>
                </svg>
              </button>
            </div>
            <div className="p-4 overflow-y-auto flex-1">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {Object.entries(strategyResults.results).map(([strategyName, result]: [string, any]) => (
                  <div key={strategyName} className="bg-[#15151e] rounded-lg p-4 border border-[#38383f]">
                    <div className="flex items-center gap-2 mb-3">
                      <div className={`w-3 h-3 rounded-full ${result.error ? 'bg-red-500' : 'bg-green-500'}`}></div>
                      <h3 className="font-bold text-white capitalize">{strategyName.replace('_', ' ')}</h3>
                      {!result.error && (
                        <span className="text-xs bg-[#e10600]/20 text-[#e10600] px-2 py-0.5 rounded">
                          {result.num_docs} docs
                        </span>
                      )}
                    </div>
                    {result.error ? (
                      <p className="text-red-400 text-sm">{result.error}</p>
                    ) : (
                      <div className="space-y-2">
                        {result.docs?.slice(0, 2).map((doc: any, i: number) => (
                          <div key={i} className="text-sm text-gray-400 bg-[#1f1f27] p-2 rounded">
                            <p className="line-clamp-3">{doc.content}</p>
                            {doc.metadata?.driver && (
                              <span className="inline-block mt-1 text-xs bg-[#38383f] px-2 py-0.5 rounded">
                                {doc.metadata.driver}
                              </span>
                            )}
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
            <div className="p-4 border-t border-[#38383f]">
              <button
                onClick={() => setShowStrategyModal(false)}
                className="f1-button-primary w-full"
              >
                Close Comparison
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
