import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router';
import { useTranslation } from 'react-i18next';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { api } from '@/lib/api';

export function Login() {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const [apiKey, setApiKey] = useState('');
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  // Check whether the browser already has a valid HttpOnly UI session.
  useEffect(() => {
    const checkAuth = async () => {
      setIsLoading(true);
      try {
        await api.getConfig();
        navigate('/dashboard');
      } catch {
        // The login form remains visible when there is no valid session.
      } finally {
        setIsLoading(false);
      }
    };

    checkAuth();
    
    // Listen for unauthorized events
    const handleUnauthorized = () => {
      navigate('/login');
    };
    
    window.addEventListener('unauthorized', handleUnauthorized);
    
    return () => {
      window.removeEventListener('unauthorized', handleUnauthorized);
    };
  }, [navigate]);

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    
    setIsLoading(true);
    setError('');
    try {
      await api.login(apiKey);
      window.dispatchEvent(new CustomEvent('authenticated'));
      navigate('/dashboard');
    } catch {
      setError(t('login.invalidApiKey'));
    } finally {
      setIsLoading(false);
    }
  };

  if (isLoading) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-background">
        <Card className="w-full max-w-md">
          <CardHeader className="space-y-1">
            <CardTitle className="text-2xl">{t('login.title')}</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex justify-center py-8">
              <div className="h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent"></div>
            </div>
            <p className="text-center text-sm text-muted-foreground">{t('login.validating')}</p>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="flex min-h-screen items-center justify-center bg-background">
      <Card className="w-full max-w-md">
        <CardHeader className="space-y-1">
          <CardTitle className="text-2xl">{t('login.title')}</CardTitle>
          <CardDescription>
            {t('login.description')}
          </CardDescription>
        </CardHeader>
        <form onSubmit={handleLogin}>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="apiKey">{t('login.apiKey')}</Label>
              <Input
                id="apiKey"
                type="password"
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                placeholder={t('login.apiKeyPlaceholder')}
              />
            </div>
            {error && <div className="text-sm text-destructive">{error}</div>}
          </CardContent>
          <CardFooter>
            <Button className="w-full" type="submit">
              {t('login.signIn')}
            </Button>
          </CardFooter>
        </form>
      </Card>
    </div>
  );
}