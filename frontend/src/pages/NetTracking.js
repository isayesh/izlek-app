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
  
  // TYT Form State
  const [tytForm, setTytForm] = useState({
    date: "",
    net_score: "",
    exam_name: ""
  });
  
  // AYT Form State
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
          'X-Firebase-UID': currentUser.uid
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
            'X-Firebase-UID': currentUser.uid
          }
        }
      );
      
      // Reset form
      setTytForm({ date: "", net_score: "", exam_name: "" });
      
      // Reload results
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
            'X-Firebase-UID': currentUser.uid
          }
        }
      );
      
      // Reset form
      setAytForm({ date: "", net_score: "", exam_name: "" });
      
      // Reload results
      await loadExamResults();
    } catch (error) {
      console.error("Error saving AYT:", error);
      alert("Kayıt sırasında hata oluştu");
    } finally {
      setSaving(false);
    }
  };

  // Separate TYT and AYT results
  const tytResults = useMemo(() => 
    examResults.filter(r => r.exam_type === "TYT").sort((a, b) => new Date(a.date) - new Date(b.date)),
    [examResults]
  );
  
  const aytResults = useMemo(() => 
    examResults.filter(r => r.exam_type === "AYT").sort((a, b) => new Date(a.date) - new Date(b.date)),
    [examResults]
  );

  // Stats calculations
  const tytStats = useMemo(() => {
    if (tytResults.length === 0) return null;
    const lastNet = tytResults[tytResults.length - 1]?.net_score || 0;
    const bestNet = Math.max(...tytResults.map(r => r.net_score));
    const last3 = tytResults.slice(-3);
    const trend = last3.length >= 2 ? 
      (last3[last3.length - 1].net_score - last3[0].net_score).toFixed(2) : 0;
    return { lastNet, bestNet, trend };
  }, [tytResults]);

  const aytStats = useMemo(() => {
    if (aytResults.length === 0) return null;
    const lastNet = aytResults[aytResults.length - 1]?.net_score || 0;
    const bestNet = Math.max(...aytResults.map(r => r.net_score));
    const last3 = aytResults.slice(-3);
    const trend = last3.length >= 2 ? 
      (last3[last3.length - 1].net_score - last3[0].net_score).toFixed(2) : 0;
    return { lastNet, bestNet, trend };
  }, [aytResults]);

  // Chart data
  const tytChartData = tytResults.map(r => ({
    date: format(parseISO(r.date), "dd MMM", { locale: tr }),
    net: r.net_score
  }));

  const aytChartData = aytResults.map(r => ({
    date: format(parseISO(r.date), "dd MMM", { locale: tr }),
    net: r.net_score
  }));

  const getTrendIcon = (trend) => {
    if (trend > 0) return <TrendingUp className="h-4 w-4 text-green-500" />;
    if (trend < 0) return <TrendingDown className="h-4 w-4 text-red-500" />;
    return <Minus className="h-4 w-4 text-gray-400" />;
  };

  if (loading) {
    return <div className="flex items-center justify-center min-h-screen">Yükleniyor...</div>;
  }

  return (
    <div className="min-h-screen p-6 md:p-10">
      {/* Header */}
      <div className="max-w-7xl mx-auto mb-12">
        <div className="flex justify-between items-center mb-8">
          <div>
            <h1 className="text-5xl font-extrabold text-gray-900 mb-3" data-testid="net-tracking-title">
              Net Takibi 📊
            </h1>
            <p className="text-lg text-gray-600 font-medium">
              TYT ve AYT netlerini takip et, gelişimini gör
            </p>
          </div>
          <div className="flex gap-3">
            <Button 
              variant="outline" 
              onClick={() => navigate("/dashboard")} 
              className="shadow-sm hover:shadow-md transition-shadow"
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
              className="shadow-sm hover:shadow-md transition-shadow"
              data-testid="btn-logout"
            >
              <LogOut className="mr-2 h-4 w-4" />
              Çıkış
            </Button>
          </div>
        </div>
      </div>

      {/* Net Entry Cards */}
      <div className="max-w-7xl mx-auto mb-10">
        <div className="grid md:grid-cols-2 gap-6">
          {/* TYT Card */}
          <Card className="shadow-lg border-2 border-blue-200" data-testid="tyt-entry-card">
            <CardHeader className="bg-gradient-to-r from-blue-50 to-blue-100">
              <CardTitle className="text-2xl font-bold text-blue-900">TYT Net Girişi</CardTitle>
            </CardHeader>
            <CardContent className="pt-6 space-y-4">
              <div>
                <Label htmlFor="tyt-date">Tarih</Label>
                <Input
                  id="tyt-date"
                  type="date"
                  value={tytForm.date}
                  onChange={(e) => setTytForm({ ...tytForm, date: e.target.value })}
                  data-testid="tyt-date-input"
                />
              </div>
              <div>
                <Label htmlFor="tyt-net">Net</Label>
                <Input
                  id="tyt-net"
                  type="number"
                  step="0.25"
                  placeholder="Örn: 85.5"
                  value={tytForm.net_score}
                  onChange={(e) => setTytForm({ ...tytForm, net_score: e.target.value })}
                  data-testid="tyt-net-input"
                />
              </div>
              <div>
                <Label htmlFor="tyt-exam-name">Deneme Adı / Yayın (Opsiyonel)</Label>
                <Input
                  id="tyt-exam-name"
                  type="text"
                  placeholder="Örn: Bilgi Sarmal 1. Deneme"
                  value={tytForm.exam_name}
                  onChange={(e) => setTytForm({ ...tytForm, exam_name: e.target.value })}
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

          {/* AYT Card */}
          <Card className="shadow-lg border-2 border-purple-200" data-testid="ayt-entry-card">
            <CardHeader className="bg-gradient-to-r from-purple-50 to-purple-100">
              <CardTitle className="text-2xl font-bold text-purple-900">AYT Net Girişi</CardTitle>
            </CardHeader>
            <CardContent className="pt-6 space-y-4">
              <div>
                <Label htmlFor="ayt-date">Tarih</Label>
                <Input
                  id="ayt-date"
                  type="date"
                  value={aytForm.date}
                  onChange={(e) => setAytForm({ ...aytForm, date: e.target.value })}
                  data-testid="ayt-date-input"
                />
              </div>
              <div>
                <Label htmlFor="ayt-net">Net</Label>
                <Input
                  id="ayt-net"
                  type="number"
                  step="0.25"
                  placeholder="Örn: 62.75"
                  value={aytForm.net_score}
                  onChange={(e) => setAytForm({ ...aytForm, net_score: e.target.value })}
                  data-testid="ayt-net-input"
                />
              </div>
              <div>
                <Label htmlFor="ayt-exam-name">Deneme Adı / Yayın (Opsiyonel)</Label>
                <Input
                  id="ayt-exam-name"
                  type="text"
                  placeholder="Örn: Karekök 3. Deneme"
                  value={aytForm.exam_name}
                  onChange={(e) => setAytForm({ ...aytForm, exam_name: e.target.value })}
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

      {/* History Section */}
      <div className="max-w-7xl mx-auto mb-10">
        <Card className="shadow-md">
          <CardHeader>
            <CardTitle className="text-2xl">Net Geçmişi</CardTitle>
          </CardHeader>
          <CardContent>
            {examResults.length === 0 ? (
              <p className="text-gray-500 text-center py-8">Henüz net kaydı yok</p>
            ) : (
              <div className="space-y-3 max-h-96 overflow-y-auto">
                {examResults.map((result) => (
                  <div 
                    key={result.id}
                    className="flex justify-between items-center p-4 bg-gray-50 rounded-lg border hover:bg-gray-100 transition-colors"
                    data-testid={`result-${result.id}`}
                  >
                    <div className="flex gap-4 items-center">
                      <span className={`font-bold text-sm px-3 py-1 rounded-full ${
                        result.exam_type === "TYT" 
                          ? "bg-blue-100 text-blue-700" 
                          : "bg-purple-100 text-purple-700"
                      }`}>
                        {result.exam_type}
                      </span>
                      <span className="text-gray-700">
                        {format(parseISO(result.date), "dd MMMM yyyy", { locale: tr })}
                      </span>
                      {result.exam_name && (
                        <span className="text-gray-500 text-sm italic">
                          {result.exam_name}
                        </span>
                      )}
                    </div>
                    <span className="text-2xl font-bold text-gray-900">
                      {result.net_score}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Analytics Section */}
      <div className="max-w-7xl mx-auto">
        <h2 className="text-3xl font-bold text-gray-900 mb-6">Analiz</h2>
        
        {/* Stats Cards */}
        <div className="grid md:grid-cols-2 gap-6 mb-8">
          {/* TYT Stats */}
          <Card className="shadow-md border-blue-200">
            <CardHeader>
              <CardTitle className="text-lg text-blue-900">TYT İstatistikleri</CardTitle>
            </CardHeader>
            <CardContent>
              {tytStats ? (
                <div className="space-y-4">
                  <div className="flex justify-between items-center">
                    <span className="text-gray-600">Son TYT:</span>
                    <span className="text-2xl font-bold text-gray-900">{tytStats.lastNet}</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-gray-600">En İyi TYT:</span>
                    <span className="text-2xl font-bold text-green-600">{tytStats.bestNet}</span>
                  </div>
                  <div className="flex justify-between items-center pt-2 border-t">
                    <span className="text-gray-600">Son 3 Deneme Trendi:</span>
                    <div className="flex items-center gap-2">
                      {getTrendIcon(tytStats.trend)}
                      <span className={`font-bold ${
                        tytStats.trend > 0 ? "text-green-600" : 
                        tytStats.trend < 0 ? "text-red-600" : "text-gray-600"
                      }`}>
                        {tytStats.trend > 0 ? "+" : ""}{tytStats.trend}
                      </span>
                    </div>
                  </div>
                </div>
              ) : (
                <p className="text-gray-500 text-center py-4">Henüz TYT verisi yok</p>
              )}
            </CardContent>
          </Card>

          {/* AYT Stats */}
          <Card className="shadow-md border-purple-200">
            <CardHeader>
              <CardTitle className="text-lg text-purple-900">AYT İstatistikleri</CardTitle>
            </CardHeader>
            <CardContent>
              {aytStats ? (
                <div className="space-y-4">
                  <div className="flex justify-between items-center">
                    <span className="text-gray-600">Son AYT:</span>
                    <span className="text-2xl font-bold text-gray-900">{aytStats.lastNet}</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-gray-600">En İyi AYT:</span>
                    <span className="text-2xl font-bold text-green-600">{aytStats.bestNet}</span>
                  </div>
                  <div className="flex justify-between items-center pt-2 border-t">
                    <span className="text-gray-600">Son 3 Deneme Trendi:</span>
                    <div className="flex items-center gap-2">
                      {getTrendIcon(aytStats.trend)}
                      <span className={`font-bold ${
                        aytStats.trend > 0 ? "text-green-600" : 
                        aytStats.trend < 0 ? "text-red-600" : "text-gray-600"
                      }`}>
                        {aytStats.trend > 0 ? "+" : ""}{aytStats.trend}
                      </span>
                    </div>
                  </div>
                </div>
              ) : (
                <p className="text-gray-500 text-center py-4">Henüz AYT verisi yok</p>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Charts */}
        <div className="grid md:grid-cols-2 gap-6">
          {/* TYT Chart */}
          <Card className="shadow-md">
            <CardHeader>
              <CardTitle className="text-lg">TYT Net Grafiği</CardTitle>
            </CardHeader>
            <CardContent>
              {tytChartData.length > 0 ? (
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={tytChartData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="date" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Line 
                      type="monotone" 
                      dataKey="net" 
                      stroke="#3b82f6" 
                      strokeWidth={2}
                      name="TYT Net"
                    />
                  </LineChart>
                </ResponsiveContainer>
              ) : (
                <p className="text-gray-500 text-center py-16">Henüz TYT verisi yok</p>
              )}
            </CardContent>
          </Card>

          {/* AYT Chart */}
          <Card className="shadow-md">
            <CardHeader>
              <CardTitle className="text-lg">AYT Net Grafiği</CardTitle>
            </CardHeader>
            <CardContent>
              {aytChartData.length > 0 ? (
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={aytChartData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="date" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Line 
                      type="monotone" 
                      dataKey="net" 
                      stroke="#a855f7" 
                      strokeWidth={2}
                      name="AYT Net"
                    />
                  </LineChart>
                </ResponsiveContainer>
              ) : (
                <p className="text-gray-500 text-center py-16">Henüz AYT verisi yok</p>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
