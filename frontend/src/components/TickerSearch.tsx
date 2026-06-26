import React, { useState, useEffect, useRef } from 'react';
import api from '../api';

interface TickerSearchProps {
  value: string;
  onChange: (val: string) => void;
  onSelect?: (val: string) => void;
  placeholder?: string;
  style?: React.CSSProperties;
  className?: string;
  mode?: string;
}

export const TickerSearch: React.FC<TickerSearchProps> = ({
  value,
  onChange,
  onSelect,
  placeholder,
  style,
  className,
  mode = 'equity'
}) => {
  const [results, setResults] = useState<any[]>([]);
  const [defaultResults, setDefaultResults] = useState<any[]>([]);
  const [showDropdown, setShowDropdown] = useState(false);
  const [isSearching, setIsSearching] = useState(false);
  const [activeIndex, setActiveIndex] = useState(-1);
  const [searchQuery, setSearchQuery] = useState('');
  
  const dropdownRef = useRef<HTMLDivElement>(null);
  const activeItemRef = useRef<HTMLDivElement>(null);
  const debounceRef = useRef<any>(null);

  // Synchronize internal searchQuery with external value for typing
  useEffect(() => {
    setSearchQuery(value);
  }, [value]);

  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        setShowDropdown(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // Fetch default tickers on mount or mode change so they are ready instantly
  useEffect(() => {
    const fetchDefault = async () => {
      try {
        const res = await api.get(`/tickers?mode=${mode}`);
        if (res.data && res.data.tickers) {
          setDefaultResults(res.data.tickers);
          setResults(res.data.tickers);
        }
      } catch (err) {
        console.error("Error fetching default tickers:", err);
      }
    };
    fetchDefault();
  }, [mode]);

  // Scroll active item into view
  useEffect(() => {
    if (activeIndex >= 0 && activeItemRef.current) {
      activeItemRef.current.scrollIntoView({
        block: 'nearest',
        behavior: 'smooth'
      });
    }
  }, [activeIndex]);

  // Reset activeIndex when results change
  useEffect(() => {
    setActiveIndex(results.length > 0 ? 0 : -1);
  }, [results]);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = e.target.value;
    setSearchQuery(val);
    onChange(val);
    
    if (debounceRef.current) clearTimeout(debounceRef.current);
    
    if (val.trim().length === 0) {
      setResults(defaultResults);
      setShowDropdown(true);
      setIsSearching(false);
      return;
    }

    setIsSearching(true);
    debounceRef.current = setTimeout(async () => {
      try {
        const res = await api.get(`/tickers?q=${encodeURIComponent(val)}&mode=${mode}`);
        if (res.data && res.data.tickers) {
          setResults(res.data.tickers);
          setShowDropdown(true);
        }
      } catch (err) {
        console.error("Error fetching tickers:", err);
      } finally {
        setIsSearching(false);
      }
    }, 200); // 200ms debounce
  };

  const handleSelect = (ticker: any) => {
    const symbol = ticker.symbol;
    onChange(symbol);
    setSearchQuery(symbol);
    setShowDropdown(false);
    if (onSelect) onSelect(symbol);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (!showDropdown) {
      if (e.key === 'Enter') {
        if (onSelect) onSelect(value);
      }
      return;
    }

    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setActiveIndex((prev) => (results.length === 0 ? -1 : (prev + 1) % results.length));
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setActiveIndex((prev) => (results.length === 0 ? -1 : (prev - 1 + results.length) % results.length));
    } else if (e.key === 'Enter') {
      e.preventDefault();
      if (activeIndex >= 0 && activeIndex < results.length) {
        handleSelect(results[activeIndex]);
      } else {
        setShowDropdown(false);
        if (onSelect) onSelect(value);
      }
    } else if (e.key === 'Escape') {
      e.preventDefault();
      setShowDropdown(false);
    }
  };

  // Helper to escape regex special characters
  const escapeRegExp = (str: string) => {
    return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  };

  // Helper to highlight matching text
  const highlightMatch = (text: string, query: string) => {
    if (!query) return <span>{text}</span>;
    // Strip exchange prefix if querying just the symbol part
    let queryClean = query.trim();
    if (queryClean.includes(':')) {
      queryClean = queryClean.split(':', 2)[1];
    }
    if (!queryClean) return <span>{text}</span>;

    const parts = text.split(new RegExp(`(${escapeRegExp(queryClean)})`, 'gi'));
    return (
      <span>
        {parts.map((part, i) => 
          part.toLowerCase() === queryClean.toLowerCase() ? (
            <strong key={i} style={{ color: '#8ea6ba', textShadow: '0 0 8px rgba(142, 166, 186, 0.4)', fontWeight: 700 }}>
              {part}
            </strong>
          ) : (
            <span key={i}>{part}</span>
          )
        )}
      </span>
    );
  };

  const getExchangeBadgeClass = (exchange: string) => {
    const ex = exchange.toLowerCase();
    if (ex === 'nse') return 'exchange-badge nse';
    if (ex === 'bse') return 'exchange-badge bse';
    if (ex === 'us') return 'exchange-badge us';
    if (ex === 'crypto') return 'exchange-badge crypto';
    if (ex === 'forex') return 'exchange-badge forex';
    return 'exchange-badge';
  };

  const parseSymbol = (fullSymbol: string) => {
    if (fullSymbol.includes(':')) {
      const parts = fullSymbol.split(':', 2);
      return { exchange: parts[0], symbol: parts[1] };
    }
    if (mode === 'crypto') {
      return { exchange: 'CRYPTO', symbol: fullSymbol };
    }
    if (mode === 'forex') {
      return { exchange: 'FOREX', symbol: fullSymbol };
    }
    return { exchange: 'NSE', symbol: fullSymbol };
  };

  return (
    <div 
      className="ticker-search-wrapper" 
      style={{ width: style?.width || '100%' }} 
      ref={dropdownRef}
    >
      <div className="ticker-search-input-container">
        <input
          type="text"
          value={searchQuery}
          onChange={handleInputChange}
          onKeyDown={handleKeyDown}
          placeholder={placeholder || "Search Symbol..."}
          className={className || "ticker-search-input"}
          style={{
            ...style,
            width: '100%', // Override width to fill container
          }}
          onFocus={() => {
            setShowDropdown(true);
          }}
        />
        <div className="ticker-search-icon">
          {isSearching ? (
            <span className="spinner" style={{ width: 14, height: 14, borderTopColor: '#6b868e' }} />
          ) : (
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="11" cy="11" r="8" />
              <line x1="21" y1="21" x2="16.65" y2="16.65" />
            </svg>
          )}
        </div>
      </div>
      
      {showDropdown && (
        <div className="ticker-search-dropdown search-dropdown-anim">
          {results.length > 0 ? (
            results.map((t, idx) => {
              const { exchange, symbol } = parseSymbol(t.symbol);
              const isActive = idx === activeIndex;
              return (
                <div 
                  key={t.symbol} 
                  ref={isActive ? activeItemRef : null}
                  onClick={() => handleSelect(t)}
                  className={`ticker-search-item ${isActive ? 'active' : ''}`}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '16px' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <span className={getExchangeBadgeClass(exchange)}>{exchange}</span>
                      <span style={{ color: '#f8fafc', fontSize: '13px', fontWeight: 600, letterSpacing: '0.3px' }}>
                        {highlightMatch(symbol, searchQuery)}
                      </span>
                    </div>
                    {t.sector && (
                      <span className="sector-badge">
                        {t.sector}
                      </span>
                    )}
                  </div>
                  <div style={{ color: '#94a3b8', fontSize: '11px', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', paddingLeft: '42px' }}>
                    {highlightMatch(t.name, searchQuery)}
                  </div>
                </div>
              );
            })
          ) : (
            searchQuery.trim().length > 0 && !isSearching && (
              <div className="no-results">No tickers found for "{searchQuery}"</div>
            )
          )}
        </div>
      )}
    </div>
  );
};
