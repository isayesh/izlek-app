import { useEffect, useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { useAuth } from '@/contexts/AuthContext';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Card, CardHeader, CardTitle, CardContent, CardFooter } from '@/components/ui/card';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { LogIn, AlertCircle } from 'lucide-react';

export default function Login() {
  const navigate = useNavigate();
  const { currentUser, login, firebaseConfigured } = useAuth();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (currentUser) {
      navigate('/dashboard');
    }
  }, [currentUser, navigate]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!email || !password) {
      setError('Lütfen tüm alanları doldurun');
      return;
    }

    try {
      setError('');
      setLoading(true);
      await login(email, password);
      navigate('/dashboard');
    } catch (error) {
      console.error('Login error:', error);
      
      // Friendly error messages in Turkish
      if (error.code === 'auth/user-not-found') {
        setError('Bu email ile kayıtlı kullanıcı bulunamadı');
      } else if (error.code === 'auth/wrong-password') {
        setError('Hatalı şifre');
      } else if (error.code === 'auth/invalid-email') {
        setError('Geçersiz email adresi');
      } else if (error.code === 'auth/too-many-requests') {
        setError('Çok fazla deneme. Lütfen daha sonra tekrar deneyin');
      } else {
        setError(error.message || 'Giriş yapılırken bir hata oluştu');
      }
    } finally {
      setLoading(false);
    }
  };

  // Show Firebase configuration warning
  if (!firebaseConfigured) {
    return (
      <div className="min-h-screen flex items-center justify-center p-6 bg-gradient-to-br from-slate-50 via-white to-blue-50 dark:from-slate-900 dark:via-slate-800 dark:to-slate-900">
        <Card className="w-full max-w-md shadow-lg dark:bg-slate-800">
          <CardHeader>
            <CardTitle className="text-2xl text-center flex items-center justify-center gap-2">
              <AlertCircle className="h-6 w-6 text-amber-500" />
              Yapılandırma Eksik
            </CardTitle>
          </CardHeader>
          <CardContent>
            <Alert className="border-amber-200 bg-amber-50">
              <AlertDescription className="text-gray-700">
                Firebase yapılandırması eksik. Lütfen <code className="bg-gray-100 px-2 py-1 rounded text-sm">.env</code> dosyanıza Firebase anahtarlarınızı ekleyin.
              </AlertDescription>
            </Alert>
            <div className="mt-6 p-4 bg-gray-50 rounded-lg border">
              <p className="text-sm font-semibold text-gray-700 mb-2">Gerekli değişkenler:</p>
              <ul className="text-xs text-gray-600 space-y-1 font-mono">
                <li>• REACT_APP_FIREBASE_API_KEY</li>
                <li>• REACT_APP_FIREBASE_AUTH_DOMAIN</li>
                <li>• REACT_APP_FIREBASE_PROJECT_ID</li>
                <li>• REACT_APP_FIREBASE_STORAGE_BUCKET</li>
                <li>• REACT_APP_FIREBASE_MESSAGING_SENDER_ID</li>
                <li>• REACT_APP_FIREBASE_APP_ID</li>
              </ul>
            </div>
          </CardContent>
          <CardFooter>
            <Button 
              onClick={() => navigate('/')} 
              variant="outline" 
              className="w-full"
              data-testid="btn-back-home"
            >
              Ana Sayfaya Dön
            </Button>
          </CardFooter>
        </Card>
      </div>
    );
  }

  return (
    <div className="min-h-screen flex items-center justify-center p-6 bg-gradient-to-br from-slate-50 via-white to-blue-50 dark:from-slate-900 dark:via-slate-800 dark:to-slate-900">
      <Card className="w-full max-w-md shadow-lg dark:bg-slate-800" data-testid="login-card">
        <CardHeader>
          <CardTitle className="text-3xl text-center font-bold text-gray-900 dark:text-white">
            Giriş Yap
          </CardTitle>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-4">
            {error && (
              <Alert variant="destructive" data-testid="login-error">
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}
            
            <div className="space-y-2">
              <Label htmlFor="email" className="dark:text-gray-200">Email</Label>
              <Input
                id="email"
                type="email"
                placeholder="ornek@email.com"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                disabled={loading}
                data-testid="input-email"
                className="dark:bg-slate-700 dark:text-white dark:border-slate-600"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="password" className="dark:text-gray-200">Şifre</Label>
              <Input
                id="password"
                type="password"
                placeholder="••••••••"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                disabled={loading}
                data-testid="input-password"
                className="dark:bg-slate-700 dark:text-white dark:border-slate-600"
              />
            </div>

            <Button 
              type="submit" 
              className="w-full" 
              disabled={loading}
              data-testid="btn-login-submit"
            >
              <LogIn className="mr-2 h-4 w-4" />
              {loading ? 'Giriş yapılıyor...' : 'Giriş Yap'}
            </Button>
          </form>
        </CardContent>
        <CardFooter className="flex flex-col space-y-4">
          <div className="text-sm text-center text-gray-600 dark:text-gray-300">
            Hesabın yok mu?{' '}
            <Link to="/register" className="text-indigo-600 dark:text-indigo-400 hover:underline font-medium" data-testid="link-register">
              Kayıt Ol
            </Link>
          </div>
          <Button 
            onClick={() => navigate('/')} 
            variant="ghost" 
            className="w-full dark:text-gray-200 dark:hover:bg-slate-700"
            data-testid="btn-back-home"
          >
            Ana Sayfaya Dön
          </Button>
        </CardFooter>
      </Card>
    </div>
  );
}