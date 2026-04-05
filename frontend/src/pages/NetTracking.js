import { useState, useEffect, useMemo } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import axios from "axios";
import { API } from "@/App";
import { Home, LogOut, TrendingUp, TrendingDown, Minus } from "lucide-react";
import { useAuth } from "@/contexts/AuthContext";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from "recharts";
import { format, parseISO } from "date-fns";
import { tr } from "date-fns/locale";

export default function NetTracking() {
  const navigate = useNavigate();
  const { currentUser, logout } = useAuth();
  const [loading, setLoading] = useState(true);
  const [examResults, setExamResults] = useState([]);

  const [tytForm, setTytForm] = useState({
    date: "",
    net_score: "",
    exam_name: ""
  });

  const [aytForm, setAytForm] = useState({
    date: "",
    net_score: "",
    exam_name: ""
  });

  const [saving, setSaving] = useState(false);

  useEffect(() => {
    if (currentUser) {
      loadExamResults();
    }
  }, [currentUser]);

  const loadExamResults = async () => {
    try {
      setLoading(true);
      const res = await axios.get(`${API}/exams`, {
        headers: {
          "X-Firebase-UID": currentUser.uid
        }
      });
      setExamResults(res.data || []);
    } catch (error) {
      console.error("Error loading exam results:", error);
    } finally {
      setLoading(false);
    }
  };

  const handleSaveTYT = async () => {
    if (!tytForm.date || !tytForm.net_score) {
      alert("Tarih ve net alanları zorunludur");
      return;
    }

    try {
      setSaving(true);
      await axios.post(
        `${API}/exams`,
        {
          exam_type: "TYT",
          date: tytForm.date,
          net_score: parseFloat(tytForm.net_score),
          exam_name: tytForm.exam_name || null
        },
        {
          headers: {
            "X-Firebase-UID": currentUser.uid
          }
        }
      );

      setTytForm({ date: "", net_score: "", exam_name: "" });
      await loadExamResults();
    } catch (error) {
      console.error("Error saving TYT:", error);
      alert("Kayıt sırasında hata oluştu");
    } finally {
      setSaving(false);
    }
  };

  const handleSaveAYT = async () => {
    if (!aytForm.date || !aytForm.net_score) {
      alert("Tarih ve net alanları zorunludur");
      return;
    }

    try {
      setSaving(true);
      await axios.post(
        `${API}/exams`,
        {
          exam_type: "AYT",
          date: aytForm.date,
          net_score: parseFloat(aytForm.net_score),
          exam_name: aytForm.exam_name || null
        },
        {
          headers: {
            "X-Firebase-UID": currentUser.uid
          }
        }
      );

      setAytForm({ date: "", net_score: "", exam_name: "" });
      await loadExamResults();
    } catch (error) {
      console.error("Error saving AYT:", error);
      alert("Kayıt sırasında hata oluştu");
    } finally {
      setSaving(false);
    }
  };

  const tytResults = useMemo(
    () => examResults.filter((result) => result.exam_type === "TYT").sort((a, b) => new Date(a.date) - new Date(b.date)),
    [examResults]
  );

  const aytResults = useMemo(
    () => examResults.filter((result) => result.exam_type === "AYT").sort((a, b) => new Date(a.date) - new Date(b.date)),
    [examResults]
  );

  const tytStats = useMemo(() => {
    if (tytResults.length === 0) return null;
    const lastNet = tytResults[tytResults.length - 1]?.net_score || 0;
    const bestNet = Math.max(...tytResults.map((result) => result.net_score));
    const last3 = tytResults.slice(-3);
    const trend = last3.length >= 2 ? (last3[last3.length - 1].net_score - last3[0].net_score).toFixed(2) : 0;
    return { lastNet, bestNet, trend };
  }, [tytResults]);

  const aytStats = useMemo(() => {
    if (aytResults.length === 0) return null;
    const lastNet = aytResults[aytResults.length - 1]?.net_score || 0;
    const bestNet = Math.max(...aytResults.map((result) => result.net_score));
    const last3 = aytResults.slice(-3);
    const trend = last3.length >= 2 ? (last3[last3.length - 1].net_score - last3[0].net_score).toFixed(2) : 0;
    return { lastNet, bestNet, trend };
  }, [aytResults]);

  const tytChartData = tytResults.map((result) => ({
    date: format(parseISO(result.date), "dd MMM", { locale: tr }),
    net: result.net_score
  }));

  const aytChartData = aytResults.map((result) => ({
    date: format(parseISO(result.date), "dd MMM", { locale: tr }),
    net: result.net_score
  }));

  const getTrendIcon = (trend) => {
    if (trend > 0) return <TrendingUp className="h-4 w-4 text-emerald-500" />;
    if (trend < 0) return <TrendingDown className="h-4 w-4 text-rose-500" />;
    return <Minus className="h-4 w-4 text-muted-foreground" />;
  };

  const surfaceCardClass = "border-border/70 bg-card/95 shadow-[0_12px_30px_-24px_rgba(15,23,42,0.45)]";
  const inputClass = "mt-2 h-11 rounded-xl border-border/70 bg-secondary/70 shadow-none placeholder:text-muted-foreground focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-0 dark:bg-secondary/80";
  const emptyStateClass = "flex flex-col items-center justify-center rounded-2xl border border-dashed border-border/70 bg-background/60 px-6 py-12 text-center";
  const chartTheme = {
    grid: "hsl(var(--border))",
    axis: "hsl(var(--muted-foreground))",
    tooltipBackground: "hsl(var(--card))",
    tooltipBorder: "hsl(var(--border))",
    tooltipText: "hsl(var(--foreground))"
  };

  const renderEmptyState = (title, description, testId) => (
    <div className={emptyStateClass} data-testid={testId}>
      <p className="text-lg font-semibold text-foreground">{title}</p>
      <p className="mt-2 max-w-sm text-sm leading-6 text-muted-foreground">{description}</p>
    </div>
  );

  if (loading) {
    return (
      <div className="flex min-h-screen items-center justify-center px-6 py-10">
        <div className="rounded-2xl border border-border/70 bg-card px-8 py-10 text-center shadow-sm">
          <p className="font-display text-2xl font-semibold text-foreground">Net Analizi hazırlanıyor</p>
          <p className="mt-2 text-sm text-muted-foreground">Sonuçların ve grafiklerin yükleniyor.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen p-6 md:p-10">
      <div className="max-w-7xl mx-auto mb-12">
        <div className="flex items-center justify-between gap-4 mb-8">
          <div className="space-y-3">
            <h1 className="font-display text-4xl font-semibold tracking-tight text-foreground sm:text-5xl" data-testid="net-tracking-title">
              Net Takibi
            </h1>
            <p className="max-w-2xl text-base font-medium text-muted-foreground sm:text-lg">
              TYT ve AYT netlerini dashboard ile uyumlu, daha temiz bir yüzeyde takip et.
            </p>
          </div>
          <div className="flex gap-3">
            <Button
              variant="outline"
              onClick={() => navigate("/dashboard")}
              data-testid="btn-dashboard"
            >
              <Home className="h-4 w-4" />
            </Button>
            <Button
              variant="outline"
              onClick={async () => {
                await logout();
                navigate("/");
              }}
              data-testid="btn-logout"
            >
              <LogOut className="mr-2 h-4 w-4" />
              Çıkış
            </Button>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto mb-10">
        <div className="grid gap-6 md:grid-cols-2">
          <Card className={`${surfaceCardClass} border-sky-200/50 dark:border-border/70`} data-testid="tyt-entry-card">
            <CardHeader className="rounded-t-2xl border-b border-border/60 bg-sky-50/70 dark:bg-sky-500/5">
              <CardTitle className="text-xl font-semibold text-sky-950 dark:text-slate-50">TYT Net Girişi</CardTitle>
            </CardHeader>
            <CardContent className="space-y-5 pt-6">
              <div className="space-y-2">
                <Label htmlFor="tyt-date" className="text-sm font-medium text-foreground">Tarih</Label>
                <Input
                  id="tyt-date"
                  type="date"
                  value={tytForm.date}
                  onChange={(e) => setTytForm({ ...tytForm, date: e.target.value })}
                  className={inputClass}
                  data-testid="tyt-date-input"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="tyt-net" className="text-sm font-medium text-foreground">Net</Label>
                <Input
                  id="tyt-net"
                  type="number"
                  step="0.25"
                  placeholder="Örn: 85.5"
                  value={tytForm.net_score}
                  onChange={(e) => setTytForm({ ...tytForm, net_score: e.target.value })}
                  className={inputClass}
                  data-testid="tyt-net-input"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="tyt-exam-name" className="text-sm font-medium text-foreground">Deneme Adı / Yayın (Opsiyonel)</Label>
                <Input
                  id="tyt-exam-name"
                  type="text"
                  placeholder="Örn: Bilgi Sarmal 1. Deneme"
                  value={tytForm.exam_name}
                  onChange={(e) => setTytForm({ ...tytForm, exam_name: e.target.value })}
                  className={inputClass}
                  data-testid="tyt-exam-name-input"
                />
              </div>
              <Button
                onClick={handleSaveTYT}
                className="w-full"
                disabled={saving}
                data-testid="tyt-save-btn"
              >
                {saving ? "Kaydediliyor..." : "Kaydet"}
              </Button>
            </CardContent>
          </Card>

          <Card className={`${surfaceCardClass} border-indigo-200/50 dark:border-border/70`} data-testid="ayt-entry-card">
            <CardHeader className="rounded-t-2xl border-b border-border/60 bg-indigo-50/70 dark:bg-indigo-500/5">
              <CardTitle className="text-xl font-semibold text-indigo-950 dark:text-slate-50">AYT Net Girişi</CardTitle>
            </CardHeader>
            <CardContent className="space-y-5 pt-6">
              <div className="space-y-2">
                <Label htmlFor="ayt-date" className="text-sm font-medium text-foreground">Tarih</Label>
                <Input
                  id="ayt-date"
                  type="date"
                  value={aytForm.date}
                  onChange={(e) => setAytForm({ ...aytForm, date: e.target.value })}
                  className={inputClass}
                  data-testid="ayt-date-input"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="ayt-net" className="text-sm font-medium text-foreground">Net</Label>
                <Input
                  id="ayt-net"
                  type="number"
                  step="0.25"
                  placeholder="Örn: 62.75"
                  value={aytForm.net_score}
                  onChange={(e) => setAytForm({ ...aytForm, net_score: e.target.value })}
                  className={inputClass}
                  data-testid="ayt-net-input"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="ayt-exam-name" className="text-sm font-medium text-foreground">Deneme Adı / Yayın (Opsiyonel)</Label>
                <Input
                  id="ayt-exam-name"
                  type="text"
                  placeholder="Örn: Karekök 3. Deneme"
                  value={aytForm.exam_name}
                  onChange={(e) => setAytForm({ ...aytForm, exam_name: e.target.value })}
                  className={inputClass}
                  data-testid="ayt-exam-name-input"
                />
              </div>
              <Button
                onClick={handleSaveAYT}
                className="w-full"
                disabled={saving}
                data-testid="ayt-save-btn"
              >
                {saving ? "Kaydediliyor..." : "Kaydet"}
              </Button>
            </CardContent>
          </Card>
        </div>
      </div>

      <div className="max-w-7xl mx-auto mb-10">
        <Card className={surfaceCardClass}>
          <CardHeader>
            <CardTitle className="text-xl font-semibold text-foreground">Net Geçmişi</CardTitle>
          </CardHeader>
          <CardContent>
            {examResults.length === 0 ? (
              renderEmptyState(
                "Henüz veri yok",
                "İlk TYT veya AYT netini eklediğinde tüm geçmişin burada sade ve temiz bir akışta görünecek.",
                "net-history-empty-state"
              )
            ) : (
              <div className="max-h-96 space-y-3 overflow-y-auto pr-2">
                {examResults.map((result) => (
                  <div
                    key={result.id}
                    className="flex items-center justify-between rounded-2xl border border-border/70 bg-background/70 p-4 transition-shadow duration-200 hover:shadow-sm"
                    data-testid={`result-${result.id}`}
                  >
                    <div className="flex flex-wrap items-center gap-3 sm:gap-4">
                      <span
                        className={`rounded-full border px-3 py-1 text-xs font-semibold ${
                          result.exam_type === "TYT"
                            ? "border-sky-200 bg-sky-50 text-sky-700 dark:border-sky-500/20 dark:bg-sky-500/10 dark:text-sky-200"
                            : "border-indigo-200 bg-indigo-50 text-indigo-700 dark:border-indigo-500/20 dark:bg-indigo-500/10 dark:text-indigo-200"
                        }`}
                      >
                        {result.exam_type}
                      </span>
                      <span className="text-sm text-foreground/90">
                        {format(parseISO(result.date), "dd MMMM yyyy", { locale: tr })}
                      </span>
                      {result.exam_name && (
                        <span className="text-sm italic text-muted-foreground">{result.exam_name}</span>
                      )}
                    </div>
                    <span className="text-2xl font-semibold text-foreground">{result.net_score}</span>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      <div className="max-w-7xl mx-auto">
        <h2 className="mb-6 text-3xl font-semibold tracking-tight text-foreground">Analiz</h2>

        <div className="mb-8 grid gap-6 md:grid-cols-2">
          <Card className={`${surfaceCardClass} border-sky-200/50 dark:border-border/70`}>
            <CardHeader>
              <CardTitle className="text-xl font-semibold text-sky-950 dark:text-slate-50">TYT İstatistikleri</CardTitle>
            </CardHeader>
            <CardContent>
              {tytStats ? (
                <div className="space-y-4">
                  <div className="flex items-center justify-between rounded-2xl border border-border/60 bg-background/60 px-4 py-3">
                    <span className="text-sm font-medium text-muted-foreground">Son TYT</span>
                    <span className="text-2xl font-semibold text-foreground">{tytStats.lastNet}</span>
                  </div>
                  <div className="flex items-center justify-between rounded-2xl border border-border/60 bg-background/60 px-4 py-3">
                    <span className="text-sm font-medium text-muted-foreground">En İyi TYT</span>
                    <span className="text-2xl font-semibold text-emerald-600">{tytStats.bestNet}</span>
                  </div>
                  <div className="flex items-center justify-between rounded-2xl border border-border/60 bg-background/60 px-4 py-3">
                    <span className="text-sm font-medium text-muted-foreground">Son 3 Deneme Trendi</span>
                    <div className="flex items-center gap-2">
                      {getTrendIcon(tytStats.trend)}
                      <span
                        className={`font-semibold ${
                          tytStats.trend > 0
                            ? "text-emerald-600"
                            : tytStats.trend < 0
                              ? "text-rose-600"
                              : "text-muted-foreground"
                        }`}
                      >
                        {tytStats.trend > 0 ? "+" : ""}
                        {tytStats.trend}
                      </span>
                    </div>
                  </div>
                </div>
              ) : (
                renderEmptyState(
                  "Henüz veri yok",
                  "TYT sonuçların geldikçe burada son net, en iyi sonuç ve trend özetini göreceksin.",
                  "tyt-stats-empty-state"
                )
              )}
            </CardContent>
          </Card>

          <Card className={`${surfaceCardClass} border-indigo-200/50 dark:border-border/70`}>
            <CardHeader>
              <CardTitle className="text-xl font-semibold text-indigo-950 dark:text-slate-50">AYT İstatistikleri</CardTitle>
            </CardHeader>
            <CardContent>
              {aytStats ? (
                <div className="space-y-4">
                  <div className="flex items-center justify-between rounded-2xl border border-border/60 bg-background/60 px-4 py-3">
                    <span className="text-sm font-medium text-muted-foreground">Son AYT</span>
                    <span className="text-2xl font-semibold text-foreground">{aytStats.lastNet}</span>
                  </div>
                  <div className="flex items-center justify-between rounded-2xl border border-border/60 bg-background/60 px-4 py-3">
                    <span className="text-sm font-medium text-muted-foreground">En İyi AYT</span>
                    <span className="text-2xl font-semibold text-emerald-600">{aytStats.bestNet}</span>
                  </div>
                  <div className="flex items-center justify-between rounded-2xl border border-border/60 bg-background/60 px-4 py-3">
                    <span className="text-sm font-medium text-muted-foreground">Son 3 Deneme Trendi</span>
                    <div className="flex items-center gap-2">
                      {getTrendIcon(aytStats.trend)}
                      <span
                        className={`font-semibold ${
                          aytStats.trend > 0
                            ? "text-emerald-600"
                            : aytStats.trend < 0
                              ? "text-rose-600"
                              : "text-muted-foreground"
                        }`}
                      >
                        {aytStats.trend > 0 ? "+" : ""}
                        {aytStats.trend}
                      </span>
                    </div>
                  </div>
                </div>
              ) : (
                renderEmptyState(
                  "Henüz veri yok",
                  "AYT sonuçların geldikçe burada son net, en iyi sonuç ve trend özetini göreceksin.",
                  "ayt-stats-empty-state"
                )
              )}
            </CardContent>
          </Card>
        </div>

        <div className="grid gap-6 md:grid-cols-2">
          <Card className={surfaceCardClass}>
            <CardHeader>
              <CardTitle className="text-xl font-semibold text-foreground">TYT Net Grafiği</CardTitle>
            </CardHeader>
            <CardContent>
              {tytChartData.length > 0 ? (
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={tytChartData}>
                    <CartesianGrid strokeDasharray="3 3" stroke={chartTheme.grid} />
                    <XAxis dataKey="date" tick={{ fill: chartTheme.axis, fontSize: 12 }} />
                    <YAxis tick={{ fill: chartTheme.axis, fontSize: 12 }} />
                    <Tooltip
                      contentStyle={{
                        borderRadius: 16,
                        border: `1px solid ${chartTheme.tooltipBorder}`,
                        backgroundColor: chartTheme.tooltipBackground,
                        color: chartTheme.tooltipText
                      }}
                    />
                    <Legend />
                    <Line
                      type="monotone"
                      dataKey="net"
                      stroke="#38bdf8"
                      strokeWidth={2.5}
                      name="TYT Net"
                    />
                  </LineChart>
                </ResponsiveContainer>
              ) : (
                renderEmptyState(
                  "Henüz veri yok",
                  "TYT grafiği, ilk net girişinden sonra burada oluşacak.",
                  "tyt-chart-empty-state"
                )
              )}
            </CardContent>
          </Card>

          <Card className={surfaceCardClass}>
            <CardHeader>
              <CardTitle className="text-xl font-semibold text-foreground">AYT Net Grafiği</CardTitle>
            </CardHeader>
            <CardContent>
              {aytChartData.length > 0 ? (
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={aytChartData}>
                    <CartesianGrid strokeDasharray="3 3" stroke={chartTheme.grid} />
                    <XAxis dataKey="date" tick={{ fill: chartTheme.axis, fontSize: 12 }} />
                    <YAxis tick={{ fill: chartTheme.axis, fontSize: 12 }} />
                    <Tooltip
                      contentStyle={{
                        borderRadius: 16,
                        border: `1px solid ${chartTheme.tooltipBorder}`,
                        backgroundColor: chartTheme.tooltipBackground,
                        color: chartTheme.tooltipText
                      }}
                    />
                    <Legend />
                    <Line
                      type="monotone"
                      dataKey="net"
                      stroke="#818cf8"
                      strokeWidth={2.5}
                      name="AYT Net"
                    />
                  </LineChart>
                </ResponsiveContainer>
              ) : (
                renderEmptyState(
                  "Henüz veri yok",
                  "AYT grafiği, ilk net girişinden sonra burada oluşacak.",
                  "ayt-chart-empty-state"
                )
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
