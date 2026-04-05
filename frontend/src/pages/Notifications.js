import { useCallback, useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import axios from "axios";
import ThemeToggle from "@/components/ThemeToggle";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { useAuth } from "@/contexts/AuthContext";
import { API } from "@/App";
import { Check, Home, Users, X } from "lucide-react";
import { formatPublicHandle, getAvatarFallback, getPublicUsername } from "@/lib/publicProfile";

export default function Notifications() {
  const navigate = useNavigate();
  const { currentUser } = useAuth();
  const [requests, setRequests] = useState([]);
  const [loading, setLoading] = useState(true);
  const [actionRequestId, setActionRequestId] = useState("");
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");

  const authHeaders = useMemo(() => (
    currentUser?.uid ? { "X-Firebase-UID": currentUser.uid } : {}
  ), [currentUser]);

  const loadRequests = useCallback(async () => {
    if (!currentUser?.uid) {
      setLoading(false);
      return;
    }

    try {
      setLoading(true);
      const response = await axios.get(`${API}/friends/requests/incoming`, { headers: authHeaders });
      setRequests(Array.isArray(response.data) ? response.data : []);
    } catch (loadError) {
      console.error("Error loading incoming friend requests:", loadError);
      setError("Bildirimler yüklenirken bir hata oluştu.");
    } finally {
      setLoading(false);
    }
  }, [authHeaders, currentUser]);

  useEffect(() => {
    loadRequests();
  }, [loadRequests]);

  const handleRequestAction = async (requestId, action) => {
    try {
      setActionRequestId(requestId);
      setError("");
      setSuccess("");
      await axios.post(`${API}/friends/requests/${requestId}/${action}`, {}, { headers: authHeaders });
      setRequests((previousRequests) => previousRequests.filter((request) => request.id !== requestId));
      setSuccess(action === "accept" ? "Arkadaşlık isteği kabul edildi." : "Arkadaşlık isteği reddedildi.");
    } catch (actionError) {
      console.error("Error updating friend request:", actionError);
      setError(actionError.response?.data?.detail || "İstek güncellenemedi.");
    } finally {
      setActionRequestId("");
    }
  };

  return (
    <div className="min-h-screen bg-background p-6 md:p-10" data-testid="notifications-page">
      <div className="mx-auto max-w-5xl space-y-6" data-testid="notifications-page-container">
        <Card className="rounded-2xl border border-border/70 bg-card shadow-[0_18px_36px_-28px_rgba(15,23,42,0.16)]" data-testid="notifications-header-card">
          <CardContent className="p-6 sm:p-8">
            <div className="flex flex-wrap items-start justify-between gap-4" data-testid="notifications-header-content">
              <div>
                <h1 className="text-4xl font-bold tracking-tight text-slate-900 dark:text-slate-100" data-testid="notifications-title">Bildirimler</h1>
                <p className="mt-2 text-base text-slate-600 dark:text-slate-300" data-testid="notifications-subtitle">
                  Gelen arkadaşlık isteklerini buradan kabul edebilir veya reddedebilirsin.
                </p>
              </div>

              <div className="flex flex-wrap items-center gap-2" data-testid="notifications-header-actions">
                <ThemeToggle className="h-10 w-10 rounded-xl border border-border/70 bg-background text-slate-700 hover:bg-secondary dark:text-slate-100" />
                <Button variant="outline" onClick={() => navigate("/friends")} className="h-10 rounded-xl border-border/70 bg-background hover:bg-secondary" data-testid="notifications-go-friends-button">
                  <Users className="mr-2 h-4 w-4" /> Arkadaşlar
                </Button>
                <Button variant="outline" onClick={() => navigate("/dashboard")} className="h-10 rounded-xl border-border/70 bg-background hover:bg-secondary" data-testid="notifications-go-dashboard-button">
                  <Home className="mr-2 h-4 w-4" /> Dashboard
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>

        {(error || success) && (
          <div
            className={`rounded-2xl border px-4 py-3 text-sm font-medium ${error ? "border-red-200 bg-red-50 text-red-700 dark:border-red-900/60 dark:bg-red-950/30 dark:text-red-300" : "border-green-200 bg-green-50 text-green-700 dark:border-green-900/60 dark:bg-green-950/30 dark:text-green-300"}`}
            data-testid={error ? "notifications-error-message" : "notifications-success-message"}
          >
            {error || success}
          </div>
        )}

        <Card className="rounded-2xl border border-border/70 bg-card shadow-[0_18px_36px_-28px_rgba(15,23,42,0.16)]" data-testid="notifications-list-card">
          <CardHeader>
            <CardTitle className="text-xl font-bold text-slate-900 dark:text-slate-100" data-testid="notifications-list-title">Gelen İstekler</CardTitle>
          </CardHeader>
          <CardContent>
            {loading ? (
              <p className="py-10 text-center text-slate-600 dark:text-slate-300" data-testid="notifications-loading-state">Yükleniyor...</p>
            ) : requests.length === 0 ? (
              <p className="py-10 text-center text-slate-600 dark:text-slate-300" data-testid="notifications-empty-state">Bekleyen arkadaşlık isteğin bulunmuyor.</p>
            ) : (
              <div className="space-y-4" data-testid="notifications-list">
                {requests.map((request) => {
                  const handleText = formatPublicHandle(request.from_handle_display || request.from_handle);
                  return (
                    <div key={request.id} className="rounded-xl border border-border/70 bg-card p-4 shadow-sm" data-testid={`notification-request-${request.id}`}>
                      <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between" data-testid={`notification-request-content-${request.id}`}>
                        <div className="flex items-start gap-3">
                          <div className="flex h-12 w-12 items-center justify-center overflow-hidden rounded-2xl bg-slate-900 text-sm font-bold text-white" data-testid={`notification-avatar-${request.id}`}>
                            {request.from_avatar_url ? (
                              <img src={request.from_avatar_url} alt={`${getPublicUsername(request)} avatar`} className="h-full w-full object-cover" data-testid={`notification-avatar-image-${request.id}`} />
                            ) : (
                              getAvatarFallback({ username: request.from_username })
                            )}
                          </div>
                          <div className="space-y-1" data-testid={`notification-details-${request.id}`}>
                            <p className="font-semibold text-slate-900 dark:text-slate-100" data-testid={`notification-username-${request.id}`}>{request.from_username}</p>
                            {handleText && <p className="text-sm text-slate-500 dark:text-slate-400" data-testid={`notification-handle-${request.id}`}>{handleText}</p>}
                            <p className="text-sm text-slate-600 dark:text-slate-300" data-testid={`notification-message-${request.id}`}>sana arkadaşlık isteği gönderdi</p>
                          </div>
                        </div>

                        <div className="flex flex-wrap gap-2" data-testid={`notification-actions-${request.id}`}>
                          <Button
                            onClick={() => handleRequestAction(request.id, "accept")}
                            disabled={actionRequestId === request.id}
                            className="h-10 rounded-xl bg-primary text-primary-foreground hover:bg-slate-800"
                            data-testid={`notification-accept-${request.id}`}
                          >
                            <Check className="mr-2 h-4 w-4" /> Kabul Et
                          </Button>
                          <Button
                            variant="outline"
                            onClick={() => handleRequestAction(request.id, "reject")}
                            disabled={actionRequestId === request.id}
                            className="h-10 rounded-xl border-border/70 bg-background hover:bg-secondary"
                            data-testid={`notification-reject-${request.id}`}
                          >
                            <X className="mr-2 h-4 w-4" /> Reddet
                          </Button>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}