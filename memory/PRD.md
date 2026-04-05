# PRD

## Orijinal Problem Statement
Mevcut İzlek projesinde sadece landing page UI/UX refinement yapılacak. Backend'e, Firebase auth akışına, route yapısına ve çalışan akışlara dokunulmayacak. Amaç; landing page'i dashboard ile aynı tasarım evreninden gelen ama onun kopyası olmayan, daha premium, sade, rafine ve showcase/marketing odaklı bir seviyeye taşımak.

## Architecture Decisions
- Sadece `/app/frontend/src/pages/LandingPage.js` üzerinde landing odaklı görsel/refinement çalışması yapıldı.
- Backend, auth, route yapısı ve config dosyaları değiştirilmedi.
- Frontend tarafında `REACT_APP_BACKEND_URL` eksik olsa da landing render olabilsin diye güvenli fallback uygulandı; backend mantığı değiştirilmedi.
- Firebase auth tarafında sadece frontend için, preview/development ortamına özel geçici bir bypass eklendi; production auth akışı korunuyor.
- Var olan `Button`, `Card`, `ThemeToggle` bileşen dili korunarak landing sakinleştirildi.
- Renk dili nötr yüzeyler + kontrollü mavi/indigo accent yaklaşımına çekildi.
- Fazla tekrar eden section yapısı sadeleştirildi: Hero, 3 adım, online çalışma odaları, feature grid, kısa FAQ, final CTA.

## Implemented
- Hero alanı dashboard ile uyumlu ama daha marketing/showcase odaklı olacak şekilde yenilendi.
- Fazla renkli ve template hissi veren bloklar kaldırıldı; nötr kart dili ve daha sakin tipografi uygulandı.
- Online çalışma odaları bölümü premium hiyerarşiyle güçlendirildi.
- Feature grid 6 sade kart ile yeniden kurgulandı.
- FAQ kısa accordion mantığında sadeleştirildi.
- Final CTA daha kontrollü gradient ve daha rafine metin diliyle güncellendi.
- Tailwind styling pipeline restore edildi; `tailwind.config.js` ve `postcss.config.js` eklendi, mevcut `index.css` ve `index.js` hattı doğrulandı.
- Layout bug fix: hero ve room preview alanlarında spacing, stacking, text wrap ve badge taşma sorunları yapıyı bozmadan düzeltildi.
- Kullanıcı tarafından sağlanan iki gerçek ürün ekranı landing’de kullanıldı; hero ve online odalar preview alanlarından mockup UI kaldırıldı.
- `REACT_APP_BACKEND_URL` eksikken import-time crash kaldırıldı; landing page backend bağımlılığı olmadan açılabiliyor.
- Preview/development ortamı için Firebase Auth bypass uygulandı: mock kullanıcı `dev-user / test@izlek.dev / isa` otomatik yükleniyor.
- `ProtectedRoute` auth bypass sinyalini okuyacak şekilde güncellendi; login redirect olmadan korumalı sayfalar açılabiliyor.
- Landing, Login ve Register sayfaları mock kullanıcı aktifken korumalı akışa doğru yönleniyor.
- Firebase config kontrolü placeholder/init-failure durumlarını daha güvenli ayırt edecek şekilde sıkılaştırıldı.

## Prioritized Backlog
### P0
- Mevcut P0 açık iş yok; auth bypass talebi tamamlandı ve test edildi.

### P1
- İstenirse preview için yalnız UI amaçlı demo profil/program verisi düşünülerek ilk ekran daha dolu gösterilebilir.

### P2
- Landing ve dashboard arasında ortak spacing/token standardını küçük bir design utility katmanına taşı.
- FAQ ve final CTA mikro etkileşimlerini daha da rafine et.

## Next Tasks
- Kullanıcı yeni bir refinement istemezse mevcut durumda ek zorunlu iş yok.
- Gerekirse mock kullanıcı için yalnız preview tarafında demo içerik gösterme stratejisi planlanabilir.

## Latest Update
- Hero ve online odalar görselleri `object-cover` ile yeniden ayarlandı.
- Hero görseli center crop + hafif zoom ile güçlendirildi.
- Oda görseli center-right odaklı crop ile aktif chat alanını öne çıkaracak şekilde güncellendi.

- Hero ve online odalar preview alanlarında eski screenshot bağlantıları kaldırıldı; kullanıcı tarafından sağlanan final cropped assets kullanıldı.

- Image container layout fix uygulandı: hero ve room preview alanlarında fixed crop kaldırıldı; görseller tam görünür, ortalı ve kesilmeden render olacak şekilde kapsayıcılar flex + auto-height mantığına çekildi.

- Latest prepared hero/room assets uygulandı; görseller contain + hafif scale-down ile çerçeve içinde tam görünür ve daha dengeli hale getirildi.

- Tighter image frame fix uygulandı: hero ve room preview kartlarında iç padding minimuma çekildi, görseller contain korunarak hafifçe büyütüldü ve frame daha dolu hale getirildi.

- Final frame polish uygulandı: image frame iç boşlukları biraz daha azaltıldı ve görseller contain korunarak hafifçe büyütüldü.

- Hero preview balance polish yapıldı: hero screenshot biraz daha büyütüldü, iç yatay boşluk azaltıldı, dark mode frame etkisi yumuşatıldı.

- Final hero presence pass yapıldı: hero preview frame dark mode’da daha gömülü hale getirildi, screenshot daha dominant ve hafif sağa dengeli hizalı olacak şekilde ayarlandı.

- Hero dominant scale pass yapıldı: right preview daha büyük ve baskın hale getirildi, padding azaltıldı, hafif sağa hizalanarak sol metin bloğuna karşı görsel ağırlık artırıldı.

- Hero ve room preview alanları exact wrapper styling ile sadeleştirildi: 560px width, transparent wrapper, no frame, no crop, no scale, direct product screenshot görünümü.

- Dark mode hero cleanup: hero screenshot dark mode’da kaldırıldı; light mode’daki preview korunurken dark hero daha temiz ve daha intentional hale getirildi.

- Dark mode hero için sağ tarafa çok hafif mor/mavi ambient glow eklendi; görünür UI eklemeden boşluk daha premium dolduruldu.

- Dark mode hero right side için düşük opaklıkta, ağır blur uygulanmış dashboard silueti eklendi; görünür kart/screenshot olmadan ürün varlığı hissi güçlendirildi.

- Landing brand işareti minimal path/progress logo ile güncellendi; light modda gradient, dark modda beyaz versiyon kullanıldı.

- Navbar brand işaretindeki kutulu görünüm kaldırıldı; logo inline brand mark olarak sadeleştirildi.

- Landing hero stats alanındaki haftalık plan değeri 5 günden 7 güne güncellendi.

- Dashboard style alignment pass yapıldı: dominant purple azaltıldı, navy/slate ağırlıklı yüzeyler, dashboard benzeri buton/kart dili ve daha minimal accent kullanımı landing’e taşındı.

- Subtle visual richness pass yapıldı: başlıkta çok hafif navy→soft blue gradient, stat ikonlarında yumuşak accent tonları ve hero sağında düşük yoğunluklu ambient depth eklendi.

- Hero premium polish uygulandı: “sen çalışmaya odaklan” ifadesine zarif navy→soft blue gradient eklendi, hero product visual için hafif gölge ve çok düşük yoğunluklu arka plan glow eklendi.

- Premium level polish: hero gradient biraz daha görünür hale getirildi, hero glow blur azaltıldı ve primary CTA hover durumu hafifçe güçlendirildi.

- Feature kartları premium tint pass ile güncellendi: “Ürün detayı” etiketi kaldırıldı, çok hafif ton varyasyonları, güçlendirilmiş icon container ve yumuşak hover elevation eklendi.

- Online çalışma odaları bölümüne çok hafif gradient yüzey, ambient glow ve yumuşak üst/alt fade ile premium katman hissi eklendi.

- Targeted section depth rollback/apply: online rooms bölümündeki son yanlış ambient efekt geri alındı; ardından yalnız “Nasıl çalışır” ve “Online çalışma odaları” bölümlerine çok hafif yüzey derinliği uygulandı.

- Son düzeltmede sadece “Üç net adımda ritmini kur” section’ına görünür ama yumuşak yüzey derinliği uygulandı; online odalar ve diğer section’lar geri bırakıldı.

- Dashboard consistency pass yapıldı: Leaderboard, Friends, Notifications ve Profile sayfaları dashboard yüzey/token diliyle hizalandı; kartlar, butonlar, tipografi ve spacing tutarlı hale getirildi.
- Leaderboard satırları dashboard benzeri sade row yapısına çekildi ve ilk üç sıra için madalya işaretleri eklendi.

- 2026-04-03: Preview/development Firebase Auth bypass tamamlandı. Mock kullanıcı sabit olarak `uid=dev-user`, `email=test@izlek.dev`, `username=isa` ile sağlanıyor; production auth akışı korunuyor.
- 2026-04-03: Frontend smoke test + testing agent + frontend testing agent doğrulamalarında protected route erişimi başarılı bulundu; blank page veya auth redirect regresyonu görülmedi.
