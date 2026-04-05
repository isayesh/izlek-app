import { Navigate } from 'react-router-dom';
import { useAuth } from '@/contexts/AuthContext';

export default function ProtectedRoute({ children }) {
  const { authBypassEnabled, currentUser, loading, firebaseConfigured } = useAuth();

  if (loading) {
    return (
      <div
        className="flex items-center justify-center min-h-screen bg-gradient-to-br from-slate-50 via-white to-blue-50 dark:from-slate-900 dark:via-slate-800 dark:to-slate-900"
        data-testid="protected-route-loading"
      >
        <div className="text-center" data-testid="protected-route-loading-state">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600 mx-auto" data-testid="protected-route-loading-spinner"></div>
          <p className="mt-4 text-gray-600 dark:text-gray-300" data-testid="protected-route-loading-text">Yükleniyor...</p>
        </div>
      </div>
    );
  }

  if (authBypassEnabled) {
    return children;
  }

  if (!firebaseConfigured) {
    return (
      <div
        className="flex items-center justify-center min-h-screen bg-gradient-to-br from-slate-50 via-white to-blue-50 dark:from-slate-900 dark:via-slate-800 dark:to-slate-900"
        data-testid="protected-route-config-error"
      >
        <div className="text-center p-8 bg-white dark:bg-slate-800 rounded-lg shadow-lg max-w-md" data-testid="protected-route-config-error-card">
          <h2 className="text-2xl font-bold text-red-600 dark:text-red-400 mb-4" data-testid="protected-route-config-error-title">Yapılandırma Hatası</h2>
          <p className="text-gray-600 dark:text-gray-300" data-testid="protected-route-config-error-description">Firebase yapılandırması eksik. Lütfen yöneticiyle iletişime geçin.</p>
        </div>
      </div>
    );
  }

  if (!currentUser) {
    return <Navigate to="/login" replace />;
  }

  return children;
}