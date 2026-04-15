import { useNavigate } from "react-router-dom";
import { Home, MessageSquare, Users } from "lucide-react";

import ThemeToggle from "@/components/ThemeToggle";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

export default function Messages() {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen bg-background p-6 md:p-10" data-testid="messages-page">
      <div className="mx-auto max-w-5xl space-y-6" data-testid="messages-page-container">
        <Card className="rounded-2xl border border-border/70 bg-card shadow-[0_18px_36px_-28px_rgba(15,23,42,0.16)]" data-testid="messages-header-card">
          <CardContent className="p-6 sm:p-8">
            <div className="flex flex-wrap items-start justify-between gap-4" data-testid="messages-header-content">
              <div>
                <h1 className="text-4xl font-bold tracking-tight text-slate-900 dark:text-slate-100" data-testid="messages-title">DM Kutum</h1>
                <p className="mt-2 text-base text-slate-600 dark:text-slate-300" data-testid="messages-subtitle">
                  Doğrudan mesajlaşma alanına buradan erişeceksin.
                </p>
              </div>

              <div className="flex flex-wrap items-center gap-2" data-testid="messages-header-actions">
                <ThemeToggle className="h-10 w-10 rounded-xl border border-border/70 bg-background text-slate-700 hover:bg-secondary dark:text-slate-100" />
                <Button variant="outline" onClick={() => navigate("/friends")} className="h-10 rounded-xl border-border/70 bg-background hover:bg-secondary" data-testid="messages-go-friends-button">
                  <Users className="mr-2 h-4 w-4" /> Arkadaşlar
                </Button>
                <Button variant="outline" onClick={() => navigate("/dashboard")} className="h-10 rounded-xl border-border/70 bg-background hover:bg-secondary" data-testid="messages-go-dashboard-button">
                  <Home className="mr-2 h-4 w-4" /> Dashboard
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="rounded-2xl border border-border/70 bg-card shadow-[0_18px_36px_-28px_rgba(15,23,42,0.16)]" data-testid="messages-placeholder-card">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-xl font-bold text-slate-900 dark:text-slate-100" data-testid="messages-placeholder-title">
              <MessageSquare className="h-5 w-5 text-slate-800 dark:text-slate-200" /> Mesajlaşma Alanı
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="rounded-2xl border border-dashed border-border/70 bg-background/60 px-6 py-10 text-center" data-testid="messages-placeholder-state">
              <p className="text-base font-semibold text-slate-900 dark:text-slate-100">DM ekranı hazırlandı</p>
              <p className="mt-2 text-sm text-slate-600 dark:text-slate-300">
                Bu alan, kullanıcıların doğrudan mesajlarını görüntülemesi için üst navigasyondan erişilebilir hale getirildi.
              </p>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
