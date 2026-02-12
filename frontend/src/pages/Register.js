import { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { useAuth } from '@/contexts/AuthContext';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Card, CardHeader, CardTitle, CardContent, CardFooter } from '@/components/ui/card';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { UserPlus, AlertCircle } from 'lucide-react';

export default function Register() {
  const navigate = useNavigate();
  const { signup, firebaseConfigured } = useAuth();
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    // Validation
    if (!name || !email || !password || !confirmPassword) {
      setError('Lütfen tüm alanları doldurun');
      return;
    }

    if (password !== confirmPassword) {
      setError('Şifreler eşleşmiyor');
      return;
    }

    if (password.length < 6) {
      setError('Şifre en az 6 karakter olmalıdır');
      return;
    }

    try {
      setError('');
      setLoading(true);
      await signup(email, password, name);
      
      // Store user name in localStorage for app compatibility
      localStorage.setItem('userName', name);
      
      navigate('/dashboard');
    } catch (error) {
      console.error('Register error:', error);
      
      // Friendly error messages in Turkish
      if (error.code === 'auth/email-already-in-use') {
        setError('Bu email adresi zaten kullanımda');
      } else if (error.code === 'auth/invalid-email') {
        setError('Geçersiz email adresi');
      } else if (error.code === 'auth/weak-password') {
        setError('Şifre çok zayıf. Daha güçlü bir şifre kullanın');
      } else {
        setError(error.message || 'Kayıt olunurken bir hata oluştu');
      }
    } finally {
      setLoading(false);
    }
  };

  // Show Firebase configuration warning
  if (!firebaseConfigured) {
    return (
      <div className="min-h-screen flex items-center justify-center p-6 bg-gradient-to-br from-slate-50 via-white to-blue-50">
        <Card className="w-full max-w-md shadow-lg">
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
    <div className="min-h-screen flex items-center justify-center p-6 bg-gradient-to-br from-slate-50 via-white to-blue-50">
      <Card className="w-full max-w-md shadow-lg" data-testid="register-card">
        <CardHeader>
          <CardTitle className="text-3xl text-center font-bold text-gray-900">
            Kayıt Ol
          </CardTitle>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-4">
            {error && (
              <Alert variant="destructive" data-testid="register-error">
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}
            
            <div className="space-y-2">
              <Label htmlFor="name">İsim</Label>
              <Input
                id="name"
                type="text"
                placeholder="Adın Soyadın"
                value={name}
                onChange={(e) => setName(e.target.value)}
                disabled={loading}
                data-testid="input-name"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="email">Email</Label>
              <Input
                id="email"
                type="email"
                placeholder="ornek@email.com"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                disabled={loading}
                data-testid="input-email"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="password">Şifre</Label>
              <Input
                id="password"
                type="password"
                placeholder="••••••••"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                disabled={loading}
                data-testid="input-password"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="confirmPassword">Şifre Tekrar</Label>
              <Input
                id="confirmPassword"
                type="password"
                placeholder="••••••••"
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
                disabled={loading}
                data-testid="input-confirm-password"
              />
            </div>

            <Button 
              type="submit" 
              className="w-full" 
              disabled={loading}
              data-testid="btn-register-submit"
            >
              <UserPlus className="mr-2 h-4 w-4" />
              {loading ? 'Kayıt oluşturuluyor...' : 'Kayıt Ol'}
            </Button>
          </form>
        </CardContent>
        <CardFooter className="flex flex-col space-y-4">
          <div className="text-sm text-center text-gray-600">
            Zaten hesabın var mı?{' '}
            <Link to="/login" className="text-indigo-600 hover:underline font-medium" data-testid="link-login">
              Giriş Yap
            </Link>
          </div>
          <Button 
            onClick={() => navigate('/')} 
            variant="ghost" 
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