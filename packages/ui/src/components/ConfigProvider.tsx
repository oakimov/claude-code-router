import { createContext, useContext, useState, useEffect, useCallback } from 'react';
import type { ReactNode, Dispatch, SetStateAction } from 'react';
import { api } from '@/lib/api';
import type { Config } from '@/types';

interface ConfigContextType {
  config: Config | null;
  setConfig: Dispatch<SetStateAction<Config | null>>;
  error: Error | null;
}

const ConfigContext = createContext<ConfigContextType | undefined>(undefined);

// eslint-disable-next-line react-refresh/only-export-components
export function useConfig() {
  const context = useContext(ConfigContext);
  if (context === undefined) {
    throw new Error('useConfig must be used within a ConfigProvider');
  }
  return context;
}

interface ConfigProviderProps {
  children: ReactNode;
}

function normalizeConfig(data: Config): Config {
  return {
    ...data,
    LOG: typeof data.LOG === 'boolean' ? data.LOG : false,
    LOG_LEVEL: typeof data.LOG_LEVEL === 'string' ? data.LOG_LEVEL : 'debug',
    CLAUDE_PATH: typeof data.CLAUDE_PATH === 'string' ? data.CLAUDE_PATH : '',
    HOST: typeof data.HOST === 'string' ? data.HOST : '127.0.0.1',
    PORT: typeof data.PORT === 'number' ? data.PORT : 3456,
    APIKEY: typeof data.APIKEY === 'string' ? data.APIKEY : '',
    API_TIMEOUT_MS: typeof data.API_TIMEOUT_MS === 'string' ? data.API_TIMEOUT_MS : '600000',
    PROXY_URL: typeof data.PROXY_URL === 'string' ? data.PROXY_URL : '',
    transformers: Array.isArray(data.transformers) ? data.transformers : [],
    Providers: Array.isArray(data.Providers) ? data.Providers : [],
    forceUseImageAgent: typeof data.forceUseImageAgent === 'boolean' ? data.forceUseImageAgent : undefined,
    StatusLine: data.StatusLine && typeof data.StatusLine === 'object' ? {
      enabled: typeof data.StatusLine.enabled === 'boolean' ? data.StatusLine.enabled : false,
      currentStyle: typeof data.StatusLine.currentStyle === 'string' ? data.StatusLine.currentStyle : 'default',
      default: data.StatusLine.default && typeof data.StatusLine.default === 'object' && Array.isArray(data.StatusLine.default.modules) ? data.StatusLine.default : { modules: [] },
      powerline: data.StatusLine.powerline && typeof data.StatusLine.powerline === 'object' && Array.isArray(data.StatusLine.powerline.modules) ? data.StatusLine.powerline : { modules: [] }
    } : {
      enabled: false,
      currentStyle: 'default',
      default: { modules: [] },
      powerline: { modules: [] }
    },
    Router: data.Router && typeof data.Router === 'object' ? {
      default: typeof data.Router.default === 'string' ? data.Router.default : '',
      background: typeof data.Router.background === 'string' ? data.Router.background : '',
      think: typeof data.Router.think === 'string' ? data.Router.think : '',
      longContext: typeof data.Router.longContext === 'string' ? data.Router.longContext : '',
      longContextThreshold: typeof data.Router.longContextThreshold === 'number' ? data.Router.longContextThreshold : 60000,
      webSearch: typeof data.Router.webSearch === 'string' ? data.Router.webSearch : '',
      image: typeof data.Router.image === 'string' ? data.Router.image : '',
      fim: typeof data.Router.fim === 'string' ? data.Router.fim : ''
    } : {
      default: '',
      background: '',
      think: '',
      longContext: '',
      longContextThreshold: 60000,
      webSearch: '',
      image: '',
      fim: ''
    },
    CUSTOM_ROUTER_PATH: typeof data.CUSTOM_ROUTER_PATH === 'string' ? data.CUSTOM_ROUTER_PATH : ''
  };
}

export function ConfigProvider({ children }: ConfigProviderProps) {
  const [config, setConfig] = useState<Config | null>(null);
  const [error, setError] = useState<Error | null>(null);

  const fetchConfig = useCallback(async () => {
    try {
      const data = await api.getConfig();
      setConfig(normalizeConfig(data));
      setError(null);
    } catch (err) {
      setConfig(null);
      if ((err as Error).message !== 'Unauthorized') {
        setError(err as Error);
      }
    }
  }, []);

  useEffect(() => {
    fetchConfig();

    const handleAuthenticated = () => fetchConfig();
    const handleUnauthorized = () => setConfig(null);
    window.addEventListener('authenticated', handleAuthenticated);
    window.addEventListener('unauthorized', handleUnauthorized);
    return () => {
      window.removeEventListener('authenticated', handleAuthenticated);
      window.removeEventListener('unauthorized', handleUnauthorized);
    };
  }, [fetchConfig]);

  return (
    <ConfigContext.Provider value={{ config, setConfig, error }}>
      {children}
    </ConfigContext.Provider>
  );
}
