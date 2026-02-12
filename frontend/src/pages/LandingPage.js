import { useNavigate } from "react-router-dom";
import { useAuth } from "@/contexts/AuthContext";
import { useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { 
  Users, 
  BarChart3, 
  Target, 
  LogIn, 
  UserPlus, 
  Clock, 
  Brain, 
  TrendingUp, 
  CheckCircle2, 
  Star,
  MessageSquare,
  Zap,
  Award,
  BookOpen,
  Calendar,
  Timer,
  ArrowRight,
  Sparkles,
  Shield,
  Heart,
  Trophy
} from "lucide-react";

export default function LandingPage() {
  const navigate = useNavigate();
  const { currentUser, firebaseConfigured } = useAuth();

  // Redirect logged-in users to dashboard
  useEffect(() => {
    if (firebaseConfigured && currentUser) {
      navigate("/dashboard");
    }
  }, [currentUser, firebaseConfigured, navigate]);

  const features = [
    {
      icon: Target,
      title: "Kişisel Çalışma Programı",
      description: "Hedeflerine ve mevcut seviyene göre tamamen kişiselleştirilmiş, günlük çalışma programı oluştur. Her konuya ne kadar süre ayırman gerektiğini öğren.",
      color: "from-indigo-500 to-purple-500"
    },
    {
      icon: Users,
      title: "Online Çalışma Odaları",
      description: "Arkadaşlarınla veya yeni insanlarla online odalarda birlikte çalış. Motivasyonunu yüksek tut, yalnız hissetme.",
      color: "from-blue-500 to-cyan-500"
    },
    {
      icon: BarChart3,
      title: "Detaylı İlerleme Takibi",
      description: "Günlük, haftalık ve aylık hedeflerini görselleştir. Hangi konularda ilerilediğini, hangilerinde daha fazla çalışman gerektiğini gör.",
      color: "from-purple-500 to-pink-500"
    },
    {
      icon: Timer,
      title: "Odaklı Çalışma Modu",
      description: "Dikkatini dağıtmadan, kendi ritmine uygun şekilde çalış. Zamanı sen yönet, odağını kaybetme.",
      color: "from-orange-500 to-red-500"
    },
    {
      icon: MessageSquare,
      title: "Anlık Sohbet",
      description: "Çalışma odalarında arkadaşlarınla anlık mesajlaşma. Soru sor, motivasyon paylaş, birbirinizi destekleyin.",
      color: "from-green-500 to-teal-500"
    },
    {
      icon: Calendar,
      title: "Akıllı Planlama",
      description: "Sınav tarihine göre otomatik olarak konuları planla. Hangi konuyu ne zaman çalışman gerektiği hep senin elinde.",
      color: "from-yellow-500 to-orange-500"
    }
  ];

  const benefits = [
    "Stres seviyeni azalt, kontrolü ele al",
    "Zaman kaybını minimize et",
    "Net hedeflerle çalış",
    "Arkadaşlarınla motive ol",
    "İlerlemeyi görsel olarak takip et",
    "Düzenli ve planlı çalış"
  ];

  const testimonials = [
    {
      name: "Ayşe K.",
      role: "Boğaziçi Üniversitesi",
      content: "İzlek app sayesinde neyi ne zaman çalışacağım konusunda hiç tereddüt yaşamadım. Program benim için otomatik planladı, ben sadece çalışmaya odaklandım.",
      rating: 5
    },
    {
      name: "Mehmet D.",
      role: "İTÜ",
      content: "Online odalarda arkadaşlarımla çalışmak motivasyonumu çok artırdı. Yalnız hissetmiyorsun, herkes aynı hedefe koşuyor.",
      rating: 5
    },
    {
      name: "Zeynep A.",
      role: "Hacettepe Üniversitesi",
      content: "İlerleme takibi özelliği harika! Hangi konuda geri kaldığımı hemen görebiliyorum ve ona göre planımı düzenliyorum.",
      rating: 5
    }
  ];

  const faqs = [
    {
      question: "İzlek app gerçekten ücretsiz mi?",
      answer: "Evet, %100 ücretsiz! Kredi kartı bilgisi gerektirmeden hemen başlayabilirsin. Hiçbir ücret ödemene gerek yok."
    },
    {
      question: "Program manasında ne gibi faydanız oluyor öğrencilere?",
      answer: "Sizi tanımak için kayıt aşamasında birkaç soru soruyoruz ve size bir program şablonu çıkarıyoruz, sizler dolduruyorsunuz."
    },
    {
      question: "Online çalışma odaları nasıl çalışıyor?",
      answer: "İstediğin bir odaya katılabilir veya kendi odanı oluşturabilirsin. Odadaki herkes odaklı çalışma modu ile senkronize çalışıyor. Anlık mesajlaşma ile soru sorabilir, motivasyon paylaşabilirsin."
    },
    {
      question: "Mobil cihazlardan kullanabilir miyim?",
      answer: "Evet! İzlek app tüm cihazlarda sorunsuz çalışıyor. Bilgisayar, tablet veya telefondan istediğin yerden erişebilirsin."
    },
    {
      question: "Program ne kadar sürede hazırlanıyor?",
      answer: "Kayıt olduktan hemen sonra 2-3 dakika içinde kişisel çalışma programın hazır! Hedeflerini gir, programını al ve hemen çalışmaya başla."
    },
    {
      question: "İstediğim zaman değişiklik yapabilir miyim?",
      answer: "Kesinlikle! İstediğin zaman hedeflerini güncelleyebilir, konuları değiştirebilir, çalışma sürelerini ayarlayabilirsin. Tamamen senin kontrolünde."
    }
  ];

  const stats = [
    { value: "100+", label: "Aktif Öğrenci" },
    { value: "5000+", label: "Tamamlanan Çalışma Saati" },
    { value: "95%", label: "Kullanıcı Memnuniyeti" },
    { value: "50+", label: "Aktif Çalışma Odası" }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-blue-50">
      {/* Navigation Bar */}
      <nav className="sticky top-0 z-50 backdrop-blur-xl bg-white/70 border-b border-gray-200/50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center space-x-2">
              <Brain className="h-8 w-8 text-indigo-600" />
              <span className="text-2xl font-bold text-gray-900" data-testid="app-logo">
                İzlek              </span>
            </div>
            <div className="flex items-center space-x-3">
              <Button
                variant="ghost"
                className="text-gray-700 hover:text-indigo-600 font-medium"
                onClick={() => navigate("/login")}
                data-testid="nav-btn-login"
              >
                <LogIn className="mr-2 h-4 w-4" />
                Giriş Yap
              </Button>
              <Button
                className="bg-indigo-600 hover:bg-indigo-700 text-white font-medium shadow-sm"
                onClick={() => navigate("/register")}
                data-testid="nav-btn-register"
              >
                <UserPlus className="mr-2 h-4 w-4" />
                Hemen Başla
              </Button>
            </div>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="relative overflow-hidden pt-20 pb-32 px-4 sm:px-6 lg:px-8" data-testid="hero-section">
        {/* Background decorations */}
        <div className="absolute top-0 right-0 w-96 h-96 bg-gradient-to-br from-indigo-100/40 to-purple-100/40 rounded-full blur-3xl -z-10 animate-pulse"></div>
        <div className="absolute bottom-0 left-0 w-80 h-80 bg-gradient-to-tr from-blue-100/40 to-cyan-100/40 rounded-full blur-3xl -z-10 animate-pulse" style={{ animationDelay: '1s' }}></div>
        
        <div className="max-w-7xl mx-auto">
          <div className="text-center max-w-4xl mx-auto">
            {/* Badge */}
            <Badge className="mb-6 px-4 py-2 text-sm bg-indigo-100 text-indigo-700 hover:bg-indigo-200 border-0 animate-bounce" data-testid="hero-badge" style={{ animationDuration: '2s' }}>
              <Sparkles className="w-4 h-4 mr-2" />
              Hedeflerine ulaşmak isteyenler için akıllı platform
            </Badge>

            {/* Main Headline */}
            <h1 
              className="text-5xl sm:text-6xl lg:text-7xl font-bold text-gray-900 mb-6 leading-tight tracking-tight" 
              data-testid="landing-title"
            >
              Stresinden Kurtul,
              <br />
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-indigo-600 via-purple-600 to-pink-600 animate-gradient">
                Hedefe Odaklan
              </span>
            </h1>

            {/* Subtitle */}
            <p 
              className="text-xl sm:text-2xl text-gray-600 mb-10 leading-relaxed max-w-3xl mx-auto" 
              data-testid="landing-subtitle"
            >
              Ne çalışacağını düşünmek yerine, <span className="font-semibold text-gray-900">çalışmaya odaklan</span>. 
              Hedeflerini belirle, kişisel programını oluştur, arkadaşlarınla birlikte çalış, başarıya ulaş.
            </p>

            {/* CTA Buttons */}
            <div className="flex flex-col sm:flex-row gap-4 justify-center items-center mb-12">
              <Button
                size="lg"
                className="group w-full sm:w-auto text-lg px-10 py-7 bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-700 hover:to-purple-700 text-white shadow-lg hover:shadow-xl transition-all duration-300 font-semibold hover:scale-105"
                onClick={() => navigate("/register")}
                data-testid="btn-register"
              >
                <UserPlus className="mr-2 h-5 w-5 group-hover:scale-110 transition-transform" />
                Ücretsiz Başla
                <ArrowRight className="ml-2 h-5 w-5 group-hover:translate-x-1 transition-transform" />
              </Button>
              
              <Button
                size="lg"
                variant="outline"
                className="group w-full sm:w-auto text-lg px-10 py-7 border-2 border-gray-300 text-gray-700 hover:bg-gray-50 hover:border-gray-400 transition-all duration-300 font-semibold hover:scale-105"
                onClick={() => navigate("/login")}
                data-testid="btn-login"
              >
                <LogIn className="mr-2 h-5 w-5 group-hover:scale-110 transition-transform" />
                Giriş Yap
              </Button>
            </div>

            {/* Trust indicators */}
            <div className="flex flex-wrap justify-center items-center gap-6 text-sm text-gray-600">
              <div className="flex items-center">
                <CheckCircle2 className="w-5 h-5 text-green-600 mr-2" />
                <span>Kredi kartı gerektirmez</span>
              </div>
              <div className="flex items-center">
                <CheckCircle2 className="w-5 h-5 text-green-600 mr-2" />
                <span>Anında başla</span>
              </div>
              <div className="flex items-center">
                <CheckCircle2 className="w-5 h-5 text-green-600 mr-2" />
                <span>Tamamen ücretsiz</span>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="py-20 bg-gradient-to-r from-indigo-600 via-purple-600 to-pink-600 relative overflow-hidden" data-testid="stats-section">
        <div className="absolute inset-0 opacity-20">
          <div className="absolute top-0 left-1/4 w-96 h-96 bg-white rounded-full blur-3xl"></div>
          <div className="absolute bottom-0 right-1/4 w-80 h-80 bg-white rounded-full blur-3xl"></div>
        </div>
        
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
          <div className="text-center mb-12">
            <h2 className="text-3xl md:text-4xl font-bold text-white mb-3">
              Potansiyelini harcamak istemeyen öğrencilerin platformu.
            </h2>
            <p className="text-xl text-indigo-100">
              Klişelerden bıkmış öğrenciler zaten İzlek app ile çalışıyor
            </p>
          </div>
          
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
            {stats.map((stat, index) => (
              <div key={index} className="text-center group">
                <div className="bg-white/10 backdrop-blur-sm rounded-2xl p-6 border-2 border-white/20 hover:bg-white/20 transition-all duration-300 hover:scale-105">
                  <div className="text-4xl md:text-5xl font-bold text-white mb-2 group-hover:scale-110 transition-transform">
                    {stat.value}
                  </div>
                  <div className="text-indigo-100 text-sm md:text-base font-medium">
                    {stat.label}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Problem Section */}
      <section className="py-24 px-4 sm:px-6 lg:px-8 bg-white" data-testid="problem-section">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16">
            <Badge className="mb-4 px-4 py-2 text-sm bg-red-100 text-red-700 border-0">
              <Heart className="w-4 h-4 mr-2" />
              Seni çok iyi anlıyoruz
            </Badge>
            <h2 className="text-4xl md:text-5xl font-bold text-gray-900 mb-6">
              Sınav Hazırlık Süreci <span className="text-red-600">Gerçekten Zor</span>
            </h2>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto leading-relaxed">
              Binlerce öğrenci her gün aynı zorluklarla karşılaşıyor. <span className="font-semibold text-gray-900">Sen yalnız değilsin.</span>
            </p>
          </div>

          <div className="grid md:grid-cols-3 gap-8 mb-16">
            <Card className="border-2 border-red-100 hover:border-red-200 transition-all duration-300 hover:shadow-xl group">
              <CardContent className="pt-8 pb-6 px-6">
                <div className="w-14 h-14 bg-gradient-to-br from-red-500 to-red-600 rounded-2xl flex items-center justify-center mb-6 group-hover:scale-110 transition-transform duration-300 shadow-lg">
                  <Clock className="w-7 h-7 text-white" />
                </div>
                <h3 className="text-xl font-bold text-gray-900 mb-3">
                  Zaman Yönetimi Sorunu
                </h3>
                <p className="text-gray-600 leading-relaxed">
                  "Hangi konuya ne kadar süre ayırmalıyım? Tüm konuları yetiştirebilecek miyim?" 
                  Sürekli bu soruları sorup <span className="font-semibold text-red-600">zaman kaybediyorsun</span>.
                </p>
              </CardContent>
            </Card>

            <Card className="border-2 border-orange-100 hover:border-orange-200 transition-all duration-300 hover:shadow-xl group">
              <CardContent className="pt-8 pb-6 px-6">
                <div className="w-14 h-14 bg-gradient-to-br from-orange-500 to-orange-600 rounded-2xl flex items-center justify-center mb-6 group-hover:scale-110 transition-transform duration-300 shadow-lg">
                  <Brain className="w-7 h-7 text-white" />
                </div>
                <h3 className="text-xl font-bold text-gray-900 mb-3">
                  Motivasyon Kaybı
                </h3>
                <p className="text-gray-600 leading-relaxed">
                  Evde tek başına çalışmak, özellikle zorlandığın zamanlarda motivasyonunu kırıyor. 
                  <span className="font-semibold text-orange-600"> Pes etmek istiyorsun</span>.
                </p>
              </CardContent>
            </Card>

            <Card className="border-2 border-yellow-100 hover:border-yellow-200 transition-all duration-300 hover:shadow-xl group">
              <CardContent className="pt-8 pb-6 px-6">
                <div className="w-14 h-14 bg-gradient-to-br from-yellow-500 to-yellow-600 rounded-2xl flex items-center justify-center mb-6 group-hover:scale-110 transition-transform duration-300 shadow-lg">
                  <Target className="w-7 h-7 text-white" />
                </div>
                <h3 className="text-xl font-bold text-gray-900 mb-3">
                  Plansız Çalışma
                </h3>
                <p className="text-gray-600 leading-relaxed">
                  Günlük plan yapmadan, rastgele çalışıyorsun. İlerlediğini hissetmiyorsun, 
                  bu da seni <span className="font-semibold text-yellow-600">daha da strese sokuyor</span>.
                </p>
              </CardContent>
            </Card>
          </div>

          <div className="text-center bg-gradient-to-br from-red-50 to-orange-50 rounded-3xl p-12 border-2 border-red-100">
            <div className="max-w-2xl mx-auto">
              <div className="flex items-center justify-center mb-6">
                <div className="w-16 h-16 bg-gradient-to-br from-red-500 to-orange-500 rounded-full flex items-center justify-center">
                  <Heart className="w-8 h-8 text-white" />
                </div>
              </div>
              <h3 className="text-3xl font-bold text-gray-900 mb-4">
                Artık bu sorunların bir <span className="text-transparent bg-clip-text bg-gradient-to-r from-green-600 to-emerald-600">çözümü var</span>
              </h3>
              <p className="text-xl text-gray-700 mb-6">
                İzlek app, sınav hazırlık sürecini kolaylaştırmak için tasarlandı
              </p>
              <div className="flex items-center justify-center">
                <ArrowRight className="w-8 h-8 text-green-600 animate-bounce" style={{ animationDuration: '1.5s' }} />
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Solution Section */}
      <section className="py-24 px-4 sm:px-6 lg:px-8 bg-gradient-to-br from-indigo-50 via-purple-50 to-pink-50" data-testid="solution-section">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16">
            <Badge className="mb-4 px-4 py-2 text-sm bg-green-100 text-green-700 border-0">
              <Zap className="w-4 h-4 mr-2" />
              Çözüm burada
            </Badge>
            <h2 className="text-4xl md:text-5xl font-bold text-gray-900 mb-6">
              İzlek app ile Kontrolü Ele Al
            </h2>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
              Sınav hazırlığını kolaylaştıran, seni adım adım hedefe götüren akıllı platform
            </p>
          </div>

          <div className="grid md:grid-cols-2 gap-12 items-center mb-16">
            <div>
              <h3 className="text-3xl font-bold text-gray-900 mb-6">
                Kişisel Çalışma Programın, Otomatik Oluşsun
              </h3>
              <p className="text-lg text-gray-600 mb-6 leading-relaxed">
                Hedef puanını ve mevcut seviyeni gir, geri kalanı bize bırak. 
                İzlek app, senin için en verimli çalışma programını otomatik olarak oluşturur.
              </p>
              <ul className="space-y-4">
                <li className="flex items-start">
                  <CheckCircle2 className="w-6 h-6 text-green-600 mr-3 mt-0.5 flex-shrink-0" />
                  <span className="text-gray-700">Her konuya ne kadar süre ayırman gerektiğini hesaplar</span>
                </li>
                <li className="flex items-start">
                  <CheckCircle2 className="w-6 h-6 text-green-600 mr-3 mt-0.5 flex-shrink-0" />
                  <span className="text-gray-700">Sınav tarihine göre günlük hedeflerini belirler</span>
                </li>
                <li className="flex items-start">
                  <CheckCircle2 className="w-6 h-6 text-green-600 mr-3 mt-0.5 flex-shrink-0" />
                  <span className="text-gray-700">Zayıf konularına daha fazla zaman ayırır</span>
                </li>
                <li className="flex items-start">
                  <CheckCircle2 className="w-6 h-6 text-green-600 mr-3 mt-0.5 flex-shrink-0" />
                  <span className="text-gray-700">İstediğin zaman planını güncelleyebilirsin</span>
                </li>
              </ul>
            </div>
            <div className="relative">
              <div className="absolute inset-0 bg-gradient-to-br from-indigo-400/20 to-purple-400/20 rounded-3xl blur-2xl"></div>
              <Card className="relative border-2 border-indigo-200 shadow-2xl">
                <CardContent className="p-8">
                  <div className="flex items-center justify-between mb-6">
                    <h4 className="text-2xl font-bold text-gray-900">Günlük Programın</h4>
                    <Badge className="bg-green-100 text-green-700 border-0">Bugün</Badge>
                  </div>
                  <div className="space-y-4">
                    <div className="flex items-center justify-between p-4 bg-indigo-50 rounded-xl">
                      <div className="flex items-center">
                        <BookOpen className="w-5 h-5 text-indigo-600 mr-3" />
                        <span className="font-semibold text-gray-900">Matematik - Türev</span>
                      </div>
                      <span className="text-sm font-medium text-indigo-600">90 dk</span>
                    </div>
                    <div className="flex items-center justify-between p-4 bg-purple-50 rounded-xl">
                      <div className="flex items-center">
                        <BookOpen className="w-5 h-5 text-purple-600 mr-3" />
                        <span className="font-semibold text-gray-900">Fizik - Hareket</span>
                      </div>
                      <span className="text-sm font-medium text-purple-600">60 dk</span>
                    </div>
                    <div className="flex items-center justify-between p-4 bg-pink-50 rounded-xl">
                      <div className="flex items-center">
                        <BookOpen className="w-5 h-5 text-pink-600 mr-3" />
                        <span className="font-semibold text-gray-900">Türkçe - Paragraf</span>
                      </div>
                      <span className="text-sm font-medium text-pink-600">45 dk</span>
                    </div>
                  </div>
                  <div className="mt-6 p-4 bg-gradient-to-r from-indigo-600 to-purple-600 rounded-xl text-white">
                    <div className="flex items-center justify-between">
                      <span className="font-semibold">Toplam Hedef</span>
                      <span className="text-2xl font-bold">195 dk</span>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </div>

          <div className="grid md:grid-cols-2 gap-12 items-center">
            <div className="order-2 md:order-1 relative">
              <div className="absolute inset-0 bg-gradient-to-br from-blue-400/20 to-cyan-400/20 rounded-3xl blur-2xl"></div>
              <Card className="relative border-2 border-blue-200 shadow-2xl">
                <CardContent className="p-8">
                  <div className="flex items-center justify-between mb-6">
                    <h4 className="text-2xl font-bold text-gray-900">Aktif Çalışma Odaları</h4>
                    <Badge className="bg-blue-100 text-blue-700 border-0">
                      <Users className="w-4 h-4 mr-1" />
                      Canlı
                    </Badge>
                  </div>
                  <div className="space-y-4">
                    <div className="p-4 bg-blue-50 rounded-xl">
                      <div className="flex items-center justify-between mb-2">
                        <span className="font-semibold text-gray-900">TYT Matematik Odası</span>
                        <span className="text-sm text-blue-600 font-medium">12 kişi</span>
                      </div>
                      <div className="flex items-center text-sm text-gray-600">
                        <Clock className="w-4 h-4 mr-1" />
                        Oturum 2/4
                      </div>
                    </div>
                    <div className="p-4 bg-cyan-50 rounded-xl">
                      <div className="flex items-center justify-between mb-2">
                        <span className="font-semibold text-gray-900">Fizik Çalışma Grubu</span>
                        <span className="text-sm text-cyan-600 font-medium">8 kişi</span>
                      </div>
                      <div className="flex items-center text-sm text-gray-600">
                        <Clock className="w-4 h-4 mr-1" />
                        Oturum 1/4
                      </div>
                    </div>
                    <div className="p-4 bg-teal-50 rounded-xl">
                      <div className="flex items-center justify-between mb-2">
                        <span className="font-semibold text-gray-900">AYT Edebiyat Salonu</span>
                        <span className="text-sm text-teal-600 font-medium">15 kişi</span>
                      </div>
                      <div className="flex items-center text-sm text-gray-600">
                        <Clock className="w-4 h-4 mr-1" />
                        Oturum 3/4
                      </div>
                    </div>
                  </div>
                  <Button className="w-full mt-6 bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-700 hover:to-cyan-700 text-white">
                    Bir Odaya Katıl
                  </Button>
                </CardContent>
              </Card>
            </div>
            <div className="order-1 md:order-2">
              <h3 className="text-3xl font-bold text-gray-900 mb-6">
                Yalnız Değilsin, Birlikte Çalışın
              </h3>
              <p className="text-lg text-gray-600 mb-6 leading-relaxed">
                Online çalışma odalarında arkadaşlarınla veya hedefi aynı olan yeni insanlarla birlikte çalış. 
                Motivasyonunu yüksek tut, pes etme.
              </p>
              <ul className="space-y-4">
                <li className="flex items-start">
                  <CheckCircle2 className="w-6 h-6 text-green-600 mr-3 mt-0.5 flex-shrink-0" />
                  <span className="text-gray-700">Odaklı çalışma modu ile senkronize çalış</span>
                </li>
                <li className="flex items-start">
                  <CheckCircle2 className="w-6 h-6 text-green-600 mr-3 mt-0.5 flex-shrink-0" />
                  <span className="text-gray-700">Anlık mesajlaşma ile soru sor, yardımlaş</span>
                </li>
                <li className="flex items-start">
                  <CheckCircle2 className="w-6 h-6 text-green-600 mr-3 mt-0.5 flex-shrink-0" />
                  <span className="text-gray-700">Kendi odanı oluştur veya mevcut odalara katıl</span>
                </li>
                <li className="flex items-start">
                  <CheckCircle2 className="w-6 h-6 text-green-600 mr-3 mt-0.5 flex-shrink-0" />
                  <span className="text-gray-700">Birlikte çalışmanın gücünü keşfet</span>
                </li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-24 px-4 sm:px-6 lg:px-8 bg-white" data-testid="features-section">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16">
            <Badge className="mb-4 px-4 py-2 text-sm bg-indigo-100 text-indigo-700 border-0">
              <Award className="w-4 h-4 mr-2" />
              Güçlü özellikler
            </Badge>
            <h2 className="text-4xl md:text-5xl font-bold text-gray-900 mb-6">
              İhtiyacın Olan Her Şey, Tek Platformda
            </h2>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
              Sınav hazırlığını baştan sona kolaylaştıran özelliklerle donatıldı
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
            {features.map((feature, index) => {
              const Icon = feature.icon;
              return (
                <Card 
                  key={index} 
                  className="group hover:shadow-2xl transition-all duration-300 border-2 hover:border-indigo-200 hover:-translate-y-2"
                  data-testid={`feature-card-${index}`}
                >
                  <CardContent className="pt-8 pb-6 px-6">
                    <div className={`w-14 h-14 bg-gradient-to-br ${feature.color} rounded-2xl flex items-center justify-center mb-6 group-hover:scale-110 transition-transform duration-300 shadow-lg`}>
                      <Icon className="w-7 h-7 text-white" />
                    </div>
                    <h3 className="text-xl font-bold text-gray-900 mb-3">
                      {feature.title}
                    </h3>
                    <p className="text-gray-600 leading-relaxed">
                      {feature.description}
                    </p>
                  </CardContent>
                </Card>
              );
            })}
          </div>
        </div>
      </section>

      {/* Benefits Section */}
      <section className="py-24 px-4 sm:px-6 lg:px-8 bg-gradient-to-br from-gray-50 to-slate-50" data-testid="benefits-section">
        <div className="max-w-7xl mx-auto">
          <div className="grid md:grid-cols-2 gap-16 items-center">
            <div>
              <Badge className="mb-4 px-4 py-2 text-sm bg-purple-100 text-purple-700 border-0">
                <Trophy className="w-4 h-4 mr-2" />
                Sana neler kazandıracak
              </Badge>
              <h2 className="text-4xl md:text-5xl font-bold text-gray-900 mb-6">
                Başarıya Giden Yolda Yanındayız
              </h2>
              <p className="text-lg text-gray-600 mb-8 leading-relaxed">
                İzlek app ile sınav hazırlık sürecin artık çok daha kolay, organize ve motivasyonlu geçecek.
              </p>
              <Button 
                size="lg"
                className="bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 text-white font-semibold px-8 py-6 text-lg shadow-lg hover:scale-105 transition-all duration-300"
                onClick={() => navigate("/register")}
                data-testid="btn-benefits-cta"
              >
                <Sparkles className="mr-2 h-5 w-5" />
                Şimdi Başla - Ücretsiz
              </Button>
            </div>
            <div>
              <div className="grid gap-4">
                {benefits.map((benefit, index) => (
                  <div 
                    key={index}
                    className="flex items-center p-5 bg-white rounded-2xl border-2 border-gray-200 hover:border-purple-200 hover:shadow-lg transition-all duration-300 hover:-translate-y-1"
                  >
                    <div className="w-10 h-10 bg-gradient-to-br from-purple-500 to-pink-500 rounded-xl flex items-center justify-center mr-4 flex-shrink-0">
                      <CheckCircle2 className="w-6 h-6 text-white" />
                    </div>
                    <span className="text-lg font-medium text-gray-800">{benefit}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Comparison Section */}
      <section className="py-24 px-4 sm:px-6 lg:px-8 bg-white" data-testid="comparison-section">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-16">
            <Badge className="mb-4 px-4 py-2 text-sm bg-indigo-100 text-indigo-700 border-0">
              <TrendingUp className="w-4 h-4 mr-2" />
              Fark nerede?
            </Badge>
            <h2 className="text-4xl md:text-5xl font-bold text-gray-900 mb-6">
              Geleneksel Yöntem vs İzlek            </h2>
            <p className="text-xl text-gray-600">
              Neden yüzlerce öğrenci İzlek app'ı tercih ediyor?
            </p>
          </div>

          <div className="grid md:grid-cols-2 gap-8">
            {/* Without İzlek app */}
            <Card className="border-2 border-red-200 bg-red-50/30">
              <CardContent className="pt-8 pb-6 px-6">
                <div className="text-center mb-6">
                  <div className="inline-flex items-center justify-center w-16 h-16 bg-red-100 rounded-full mb-4">
                    <span className="text-3xl">😰</span>
                  </div>
                  <h3 className="text-2xl font-bold text-gray-900 mb-2">
                    Geleneksel Yöntem
                  </h3>
                  <p className="text-sm text-gray-600">İzlek app olmadan</p>
                </div>
                <div className="space-y-4">
                  <div className="flex items-start p-4 bg-white rounded-xl border border-red-200">
                    <div className="w-6 h-6 rounded-full bg-red-500 flex items-center justify-center mr-3 flex-shrink-0 mt-0.5">
                      <span className="text-white text-xs font-bold">✕</span>
                    </div>
                    <span className="text-gray-700">Her gün "Ne çalışmalıyım?" diye stres yaşamak</span>
                  </div>
                  <div className="flex items-start p-4 bg-white rounded-xl border border-red-200">
                    <div className="w-6 h-6 rounded-full bg-red-500 flex items-center justify-center mr-3 flex-shrink-0 mt-0.5">
                      <span className="text-white text-xs font-bold">✕</span>
                    </div>
                    <span className="text-gray-700">Motivasyon kaybı ve yalnızlık hissi</span>
                  </div>
                  <div className="flex items-start p-4 bg-white rounded-xl border border-red-200">
                    <div className="w-6 h-6 rounded-full bg-red-500 flex items-center justify-center mr-3 flex-shrink-0 mt-0.5">
                      <span className="text-white text-xs font-bold">✕</span>
                    </div>
                    <span className="text-gray-700">Hangi konuda geri kaldığını bilememek</span>
                  </div>
                  <div className="flex items-start p-4 bg-white rounded-xl border border-red-200">
                    <div className="w-6 h-6 rounded-full bg-red-500 flex items-center justify-center mr-3 flex-shrink-0 mt-0.5">
                      <span className="text-white text-xs font-bold">✕</span>
                    </div>
                    <span className="text-gray-700">Plansız, rastgele çalışma</span>
                  </div>
                  <div className="flex items-start p-4 bg-white rounded-xl border border-red-200">
                    <div className="w-6 h-6 rounded-full bg-red-500 flex items-center justify-center mr-3 flex-shrink-0 mt-0.5">
                      <span className="text-white text-xs font-bold">✕</span>
                    </div>
                    <span className="text-gray-700">Sürekli zaman kaybı ve verimsize çalışma</span>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* With İzlek app */}
            <Card className="border-2 border-green-200 bg-gradient-to-br from-green-50 to-emerald-50 relative overflow-hidden">
              <div className="absolute top-4 right-4">
                <Badge className="bg-green-600 text-white hover:bg-green-700 border-0">
                  <Star className="w-3 h-3 mr-1" />
                  Önerilen
                </Badge>
              </div>
              <CardContent className="pt-8 pb-6 px-6">
                <div className="text-center mb-6">
                  <div className="inline-flex items-center justify-center w-16 h-16 bg-green-100 rounded-full mb-4">
                    <span className="text-3xl">🚀</span>
                  </div>
                  <h3 className="text-2xl font-bold text-gray-900 mb-2">
                    İzlek app ile
                  </h3>
                  <p className="text-sm text-gray-600">Akıllı, organize, başarılı</p>
                </div>
                <div className="space-y-4">
                  <div className="flex items-start p-4 bg-white rounded-xl border border-green-200 shadow-sm">
                    <div className="w-6 h-6 rounded-full bg-gradient-to-br from-green-500 to-emerald-500 flex items-center justify-center mr-3 flex-shrink-0 mt-0.5">
                      <CheckCircle2 className="w-4 h-4 text-white" />
                    </div>
                    <span className="text-gray-700 font-medium">Otomatik oluşan program şablonu</span>
                  </div>
                  <div className="flex items-start p-4 bg-white rounded-xl border border-green-200 shadow-sm">
                    <div className="w-6 h-6 rounded-full bg-gradient-to-br from-green-500 to-emerald-500 flex items-center justify-center mr-3 flex-shrink-0 mt-0.5">
                      <CheckCircle2 className="w-4 h-4 text-white" />
                    </div>
                    <span className="text-gray-700 font-medium">Arkadaşlarınla online odalarda çalışma</span>
                  </div>
                  <div className="flex items-start p-4 bg-white rounded-xl border border-green-200 shadow-sm">
                    <div className="w-6 h-6 rounded-full bg-gradient-to-br from-green-500 to-emerald-500 flex items-center justify-center mr-3 flex-shrink-0 mt-0.5">
                      <CheckCircle2 className="w-4 h-4 text-white" />
                    </div>
                    <span className="text-gray-700 font-medium">Detaylı ilerleme takibi ve görselleştirme</span>
                  </div>
                  <div className="flex items-start p-4 bg-white rounded-xl border border-green-200 shadow-sm">
                    <div className="w-6 h-6 rounded-full bg-gradient-to-br from-green-500 to-emerald-500 flex items-center justify-center mr-3 flex-shrink-0 mt-0.5">
                      <CheckCircle2 className="w-4 h-4 text-white" />
                    </div>
                    <span className="text-gray-700 font-medium">Günlük hedefler ve net yol haritası</span>
                  </div>
                  <div className="flex items-start p-4 bg-white rounded-xl border border-green-200 shadow-sm">
                    <div className="w-6 h-6 rounded-full bg-gradient-to-br from-green-500 to-emerald-500 flex items-center justify-center mr-3 flex-shrink-0 mt-0.5">
                      <CheckCircle2 className="w-4 h-4 text-white" />
                    </div>
                    <span className="text-gray-700 font-medium">Maksimum verim ve düzenli çalışma</span>
                  </div>
                </div>
                <div className="mt-6">
                  <Button
                    className="w-full bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-700 hover:to-emerald-700 text-white font-semibold shadow-lg hover:scale-105 transition-all duration-300"
                    size="lg"
                    onClick={() => navigate("/register")}
                    data-testid="btn-comparison-cta"
                  >
                    <UserPlus className="mr-2 h-5 w-5" />
                    Hemen Ücretsiz Başla
                    <ArrowRight className="ml-2 h-5 w-5" />
                  </Button>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      {/* Testimonials Section */}
      <section className="py-24 px-4 sm:px-6 lg:px-8 bg-white" data-testid="testimonials-section">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16">
            <Badge className="mb-4 px-4 py-2 text-sm bg-yellow-100 text-yellow-700 border-0">
              <Star className="w-4 h-4 mr-2" />
              Başarı hikayeleri
            </Badge>
            <h2 className="text-4xl md:text-5xl font-bold text-gray-900 mb-6">
              Yüzlerce Öğrenci İzlek app ile Başardı
            </h2>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
              Sen de onlardan biri olabilirsin
            </p>
          </div>

          <div className="grid md:grid-cols-3 gap-8">
            {testimonials.map((testimonial, index) => (
              <Card 
                key={index}
                className="border-2 border-gray-200 hover:border-yellow-200 hover:shadow-xl transition-all duration-300"
                data-testid={`testimonial-card-${index}`}
              >
                <CardContent className="pt-8 pb-6 px-6">
                  <div className="flex mb-4">
                    {[...Array(testimonial.rating)].map((_, i) => (
                      <Star key={i} className="w-5 h-5 text-yellow-400 fill-yellow-400" />
                    ))}
                  </div>
                  <p className="text-gray-700 mb-6 leading-relaxed italic">
                    "{testimonial.content}"
                  </p>
                  <div className="border-t pt-4">
                    <div className="font-bold text-gray-900">{testimonial.name}</div>
                    <div className="text-sm text-gray-600">{testimonial.role}</div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      </section>

      {/* How It Works Section */}
      <section className="py-24 px-4 sm:px-6 lg:px-8 bg-gradient-to-br from-slate-50 to-blue-50" data-testid="how-it-works-section">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16">
            <Badge className="mb-4 px-4 py-2 text-sm bg-blue-100 text-blue-700 border-0">
              <Zap className="w-4 h-4 mr-2" />
              Çok kolay
            </Badge>
            <h2 className="text-4xl md:text-5xl font-bold text-gray-900 mb-6">
              3 Adımda Başla
            </h2>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
              Karmaşık hiçbir şey yok. Dakikalar içinde çalışmaya başla.
            </p>
          </div>

          <div className="grid md:grid-cols-3 gap-12 max-w-5xl mx-auto">
            <div className="text-center relative">
              <div className="absolute top-0 left-1/2 -translate-x-1/2 -translate-y-1/2 w-16 h-16 bg-gradient-to-br from-indigo-500 to-purple-500 rounded-full flex items-center justify-center text-white text-2xl font-bold shadow-lg">
                1
              </div>
              <div className="pt-12 pb-6 px-6 bg-white rounded-3xl border-2 border-indigo-100 shadow-lg hover:shadow-xl transition-all duration-300 h-full">
                <UserPlus className="w-16 h-16 text-indigo-600 mx-auto mb-6" />
                <h3 className="text-2xl font-bold text-gray-900 mb-3">Ücretsiz Kaydol</h3>
                <p className="text-gray-600 leading-relaxed">
                  Email adresinle saniyeler içinde hesap oluştur. Kredi kartı gerektirmez.
                </p>
              </div>
            </div>

            <div className="text-center relative">
              <div className="absolute top-0 left-1/2 -translate-x-1/2 -translate-y-1/2 w-16 h-16 bg-gradient-to-br from-purple-500 to-pink-500 rounded-full flex items-center justify-center text-white text-2xl font-bold shadow-lg">
                2
              </div>
              <div className="pt-12 pb-6 px-6 bg-white rounded-3xl border-2 border-purple-100 shadow-lg hover:shadow-xl transition-all duration-300 h-full">
                <Target className="w-16 h-16 text-purple-600 mx-auto mb-6" />
                <h3 className="text-2xl font-bold text-gray-900 mb-3">Hedefini Belirle</h3>
                <p className="text-gray-600 leading-relaxed">
                  Hedef puanını ve mevcut seviyeni gir. Program otomatik oluşsun.
                </p>
              </div>
            </div>

            <div className="text-center relative">
              <div className="absolute top-0 left-1/2 -translate-x-1/2 -translate-y-1/2 w-16 h-16 bg-gradient-to-br from-pink-500 to-red-500 rounded-full flex items-center justify-center text-white text-2xl font-bold shadow-lg">
                3
              </div>
              <div className="pt-12 pb-6 px-6 bg-white rounded-3xl border-2 border-pink-100 shadow-lg hover:shadow-xl transition-all duration-300 h-full">
                <Sparkles className="w-16 h-16 text-pink-600 mx-auto mb-6" />
                <h3 className="text-2xl font-bold text-gray-900 mb-3">Çalışmaya Başla</h3>
                <p className="text-gray-600 leading-relaxed">
                  Hepsi bu! Artık odaklanıp çalışabilirsin. Geri kalanı bizde.
                </p>
              </div>
            </div>
          </div>

          <div className="text-center mt-16">
            <Button
              size="lg"
              className="bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-700 hover:to-purple-700 text-white font-semibold px-10 py-7 text-lg shadow-lg"
              onClick={() => navigate("/register")}
              data-testid="btn-how-it-works-cta"
            >
              <UserPlus className="mr-2 h-5 w-5" />
              Hemen Başla - Ücretsiz
              <ArrowRight className="ml-2 h-5 w-5" />
            </Button>
          </div>
        </div>
      </section>

      {/* FAQ Section */}
      <section className="py-24 px-4 sm:px-6 lg:px-8 bg-white" data-testid="faq-section">
        <div className="max-w-4xl mx-auto">
          <div className="text-center mb-16">
            <Badge className="mb-4 px-4 py-2 text-sm bg-indigo-100 text-indigo-700 border-0">
              <MessageSquare className="w-4 h-4 mr-2" />
              Sıkça sorulan sorular
            </Badge>
            <h2 className="text-4xl md:text-5xl font-bold text-gray-900 mb-6">
              Aklına Takılanlar
            </h2>
            <p className="text-xl text-gray-600">
              En çok merak edilen soruları yanıtladık
            </p>
          </div>

          <div className="space-y-4">
            {faqs.map((faq, index) => (
              <Card 
                key={index}
                className="border-2 border-gray-200 hover:border-indigo-200 transition-colors"
                data-testid={`faq-card-${index}`}
              >
                <CardContent className="pt-6 pb-6 px-6">
                  <h3 className="text-lg font-bold text-gray-900 mb-3 flex items-start">
                    <CheckCircle2 className="w-6 h-6 text-indigo-600 mr-3 mt-0.5 flex-shrink-0" />
                    {faq.question}
                  </h3>
                  <p className="text-gray-600 leading-relaxed ml-9">
                    {faq.answer}
                  </p>
                </CardContent>
              </Card>
            ))}
          </div>

          <div className="text-center mt-12 p-8 bg-gradient-to-br from-indigo-50 to-purple-50 rounded-3xl border-2 border-indigo-100">
            <h3 className="text-2xl font-bold text-gray-900 mb-3">
              Başka sorun mu var?
            </h3>
            <p className="text-gray-600 mb-6">
              Kayıt olduktan sonra destek ekibimize dilediğin zaman ulaşabilirsin.
            </p>
            <Button
              className="bg-indigo-600 hover:bg-indigo-700 text-white font-semibold"
              onClick={() => navigate("/register")}
              data-testid="btn-faq-cta"
            >
              Hemen Başla
            </Button>
          </div>
        </div>
      </section>

      {/* Final CTA Section */}
      <section className="py-32 px-4 sm:px-6 lg:px-8 bg-gradient-to-br from-indigo-600 via-purple-600 to-pink-600 relative overflow-hidden" data-testid="final-cta-section">
        <div className="absolute inset-0 opacity-10">
          <div className="absolute top-0 right-0 w-96 h-96 bg-white rounded-full blur-3xl animate-pulse"></div>
          <div className="absolute bottom-0 left-0 w-80 h-80 bg-white rounded-full blur-3xl animate-pulse" style={{ animationDelay: '1s' }}></div>
          <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] bg-white rounded-full blur-3xl animate-pulse" style={{ animationDelay: '2s' }}></div>
        </div>
        
        <div className="max-w-5xl mx-auto text-center relative z-10">
          <Badge className="mb-6 px-6 py-3 text-base bg-white/20 text-white border-2 border-white/30 backdrop-blur-sm hover:bg-white/30 transition-colors">
            <Sparkles className="w-5 h-5 mr-2" />
            Binlerce öğrenci zaten başladı
          </Badge>
          
          <h2 className="text-4xl sm:text-5xl md:text-6xl lg:text-7xl font-bold text-white mb-8 leading-tight">
            Başarı Hikayen <span className="block mt-2">Bugün Başlıyor</span>
          </h2>
          
          <p className="text-xl sm:text-2xl md:text-3xl text-white/90 mb-12 leading-relaxed max-w-4xl mx-auto font-medium">
            Hedeflerine ulaşmak isteyen yüzlerce öğrenci gibi sen de İzlek app'ı kullan, 
            hedefine <span className="font-bold text-white">daha organize</span> ve <span className="font-bold text-white">motive</span> bir şekilde ilerle.
          </p>
          
          <div className="flex flex-col sm:flex-row gap-5 justify-center items-center mb-16">
            <Button
              size="lg"
              className="group w-full sm:w-auto text-xl px-14 py-9 bg-white hover:bg-gray-50 text-indigo-600 shadow-2xl hover:shadow-3xl transition-all duration-300 font-bold hover:scale-110"
              onClick={() => navigate("/register")}
              data-testid="btn-final-cta"
            >
              <UserPlus className="mr-3 h-7 w-7 group-hover:scale-110 transition-transform" />
              Hemen Ücretsiz Başla
              <ArrowRight className="ml-3 h-7 w-7 group-hover:translate-x-2 transition-transform" />
            </Button>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-6 max-w-3xl mx-auto mb-12">
            <div className="flex flex-col items-center p-6 bg-white/10 backdrop-blur-sm rounded-2xl border-2 border-white/20">
              <Shield className="w-10 h-10 text-white mb-3" />
              <span className="text-white font-semibold text-lg">%100 Güvenli</span>
              <span className="text-white/80 text-sm mt-1">SSL sertifikalı</span>
            </div>
            <div className="flex flex-col items-center p-6 bg-white/10 backdrop-blur-sm rounded-2xl border-2 border-white/20">
              <CheckCircle2 className="w-10 h-10 text-white mb-3" />
              <span className="text-white font-semibold text-lg">Kredi Kartı Yok</span>
              <span className="text-white/80 text-sm mt-1">Hiçbir ücret yok</span>
            </div>
            <div className="flex flex-col items-center p-6 bg-white/10 backdrop-blur-sm rounded-2xl border-2 border-white/20">
              <Zap className="w-10 h-10 text-white mb-3" />
              <span className="text-white font-semibold text-lg">Anında Başla</span>
              <span className="text-white/80 text-sm mt-1">2 dakikada hazır</span>
            </div>
          </div>

          <div className="bg-white/10 backdrop-blur-md rounded-3xl p-8 border-2 border-white/20 max-w-3xl mx-auto">
            <h3 className="text-2xl md:text-3xl font-bold text-white mb-4">
              🎯 Özel Fırsat
            </h3>
            <p className="text-lg md:text-xl text-white/90 mb-6">
              Şu anda kayıt olan öğrenciler, <span className="font-bold text-white">tüm özelliklere</span> ücretsiz erişim kazanıyor!
            </p>
            <div className="flex flex-wrap justify-center gap-4 text-white/90 text-sm md:text-base">
              <div className="flex items-center">
                <CheckCircle2 className="w-5 h-5 mr-2 text-green-300" />
                <span>Sınırsız çalışma odası</span>
              </div>
              <div className="flex items-center">
                <CheckCircle2 className="w-5 h-5 mr-2 text-green-300" />
                <span>Gelişmiş istatistikler</span>
              </div>
              <div className="flex items-center">
                <CheckCircle2 className="w-5 h-5 mr-2 text-green-300" />
                <span>Öncelikli destek</span>
              </div>
            </div>
          </div>

          <p className="text-white/80 text-lg mt-12 font-medium">
            Daha ne bekliyorsun? 🚀
          </p>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-12 px-4 sm:px-6 lg:px-8 bg-gray-900 text-gray-300">
        <div className="max-w-7xl mx-auto">
          <div className="flex flex-col md:flex-row justify-between items-center">
            <div className="flex items-center space-x-2 mb-4 md:mb-0">
              <Brain className="h-8 w-8 text-indigo-400" />
              <span className="text-2xl font-bold text-white">İzlek app</span>
            </div>
            <div className="text-center md:text-right">
              <p className="text-sm">© İzlek app. Tüm hakları saklıdır.</p>
              <p className="text-sm mt-1">Biz sana balık vermiyoruz, balık tutmayı öğretiyoruz.</p>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
