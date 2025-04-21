'use client';

import { useState } from 'react';

interface Article {
  title: string;
  date: string;
  url: string;
}

interface SearchResponse {
  results: Article[];
  response: string;
  metadata: {
    search_used: boolean;
    num_search_results: number;
    execution_time: number;
    date_range: {
      from: string;
      to: string;
    };
  };
}

export default function Home() {
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [articles, setArticles] = useState<Article[]>([]);
  const [aiResponse, setAiResponse] = useState<string>('');
  const [metadata, setMetadata] = useState<SearchResponse['metadata'] | null>(null);

  const handleSearch = async () => {
    if (!query.trim()) return;

    setLoading(true);
    setError(null);

    try {
      const response = await fetch('/api/news', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query }),
      });

      if (!response.ok) {
        throw new Error('Failed to fetch news');
      }

      const data: SearchResponse = await response.json();
      
      const aiResult = data.response;
      setAiResponse(aiResult);

      setArticles(data.results);
      setMetadata(data.metadata);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-emerald-50 to-blue-100 dark:from-emerald-950 dark:to-blue-900">
      <div className="max-w-4xl mx-auto px-6 py-12">
        <h1 className="text-4xl font-bold mb-8 text-emerald-900 dark:text-emerald-100">
          News Search
        </h1>
        
        <div className="flex gap-4 mb-8">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
            placeholder="Enter your search query..."
            className="flex-1 p-4 rounded-2xl border-2 border-emerald-100 bg-white/80 backdrop-blur-sm 
                     shadow-lg focus:outline-none focus:border-emerald-400 transition-all duration-200
                     dark:bg-slate-900/80 dark:border-emerald-900 dark:text-white"
          />
          <button
            onClick={handleSearch}
            disabled={loading}
            className="px-8 py-4 bg-gradient-to-r from-emerald-500 to-blue-500 text-white rounded-2xl
                     font-medium shadow-lg hover:shadow-xl transition-all duration-200 
                     hover:from-emerald-600 hover:to-blue-600 disabled:opacity-50 disabled:cursor-not-allowed
                     dark:from-emerald-600 dark:to-blue-600 dark:hover:from-emerald-500 dark:hover:to-blue-500"
          >
            {loading ? 'Searching...' : 'Search'}
          </button>
        </div>

        {error && (
          <div className="mb-8 p-6 bg-red-50 border-l-4 border-red-500 rounded-r-2xl text-red-700 dark:bg-red-950 dark:text-red-200">
            {error}
          </div>
        )}

        {aiResponse && (
          <div className="mb-8 p-8 bg-white/80 backdrop-blur-sm rounded-3xl shadow-xl border-2 border-emerald-100
                        dark:bg-slate-900/80 dark:border-emerald-900">
            <h2 className="text-2xl font-semibold mb-4 text-emerald-900 dark:text-emerald-100">AI Summary</h2>
            <p className="text-gray-700 whitespace-pre-wrap leading-relaxed dark:text-gray-300">{aiResponse}</p>
          </div>
        )}

        {articles.length > 0 && (
          <div className="space-y-4">
            <h2 className="text-2xl font-semibold mb-6 text-emerald-900 dark:text-emerald-100">Sources</h2>
            <div className="grid gap-4">
              {articles.map((article, index) => {
                // Extract domain from URL only if it's a valid URL
                let domain = '';
                if (article.url && article.url !== '#') {
                  try {
                    domain = new URL(article.url).hostname.replace('www.', '');
                  } catch (e) {
                    console.warn('Invalid URL:', article.url);
                  }
                }
                
                return (
                  <div key={index} 
                       className="p-6 bg-white/80 backdrop-blur-sm rounded-2xl shadow-lg border border-emerald-100
                                hover:shadow-xl transition-all duration-200
                                dark:bg-slate-900/80 dark:border-emerald-900">
                    <h3 className="font-medium mb-2">
                      <a 
                        href={article.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-emerald-600 hover:text-emerald-700 dark:text-emerald-400 dark:hover:text-emerald-300"
                      >
                        {article.title}
                      </a>
                    </h3>
                    <div className="flex items-center gap-3 text-sm">
                      <span className="text-emerald-700 dark:text-emerald-500">
                        {article.date}
                      </span>
                      {domain && (
                        <>
                          <span className="text-emerald-600/60 dark:text-emerald-400/60">•</span>
                          <span className="text-emerald-600 dark:text-emerald-400 font-medium">
                            {domain}
                          </span>
                        </>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}