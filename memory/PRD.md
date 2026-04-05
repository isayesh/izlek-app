# İzlek Onboarding Refresh PRD

## Original Problem Statement
Mevcut İzlek kod tabanında sadece `/program/create` onboarding ekranı minimal ve production-safe şekilde iyileştirildi. Hedefler: onboarding akışını 5 adıma taşımak, mevcut dashboard design language’ına yaklaştırmak, dark mode contrast problemlerini gidermek, radio UI’ları modern selectable card yapısına çevirmek, route/auth/backend akışını bozmamak.

## Architecture Decisions
- Sadece `frontend/src/pages/ProgramCreation.js` üzerinde uygulama değişikliği yapıldı.
- Backend endpoint sözleşmeleri korunarak onboarding submit akışı mevcut `/api/profiles` ve `/api/programs` endpoint’lerini kullanmaya devam etti.
- Kullanıcıya görünmeyen zorunlu backend alanları için güvenli varsayılanlar korundu: `exam_goal=TYT`, `daily_hours=2-4`.
- `study_field` mapping korundu; UI’da `Eşit Ağırlık` gösterilirken backend-safe değer `EA` gönderiliyor. Yeni `Dil` seçeneği eklendi.
- Yeni `class_status` sorusu backend sözleşmesini bozmamak için yalnızca onboarding state/draft içinde tutuldu; backend refactor yapılmadı.
- Draft autosave yapısı korundu, eski draft’lar için versiyonlu geri uyumluluk eklendi.

## Implemented
- RoomPage `Katılımcılar` bileşeni ölçeklenebilir hale getirildi: ilk görünümde en fazla 2 kişi gösteriliyor, fazlası için `+X kişi daha · Tümünü gör` CTA’sı ile modal açılıyor.
- Katılımcılar kartındaki iç scroll kaldırıldı; avatar/initial, kullanıcı adı, alan bilgisi ve sahip etiketi korunarak tam liste modal içinde gösteriliyor.
- Dashboard üst özet kartları profile’ın yeni modeline bağlandı: eski `Hedef` ve `Günlük tempo` kartları kaldırıldı, yerine `grade_level + study_field` tabanlı koşullu profil özeti kartı eklendi.
- Dashboard fallback davranışı sadeleştirildi: grade/alan birlikteyse tek satır `Mezun • Sayısal`, biri varsa tek başına, ikisi de yoksa kart hiç render edilmiyor.
- Profile sayfasında `study_goal` ve `daily_study_hours` inputları UI’dan kaldırıldı; yerine `grade_level` ve editable `study_field` alanları eklendi.
- Profile preview güncellendi: `@handle`, `username`, `email` altında grade + alan tek satırda gösteriliyor; eksik veri durumunda satır otomatik sadeleşiyor veya tamamen gizleniyor.
- Backend profile sözleşmesi minimal genişletildi: `grade_level` eklendi, simple profile update akışı `study_field` düzenlemeyi destekliyor ve UI artık legacy alanları göndermese de `study_goal` / `daily_study_hours` korunuyor.
- Room create/join formlarındaki alan seçimi tamamen kaldırıldı; create ve join payload’larından study_field bağımlılığı temizlendi.
- Room participant ve chat identity’lerinde alan bilgisi artık `profiles.study_field` üzerinden resolve ediliyor; profile’da alan yoksa UI bu metni sessizce gizliyor.
- Leaderboard tarafında alan gösterimi olmadığı için ek UI değişikliği yapılmadı; mevcut görünüm korunarak regresyon doğrulandı.
- Rooms create formundan `Alan (Opsiyonel)` alanı tamamen kaldırıldı; ilgili create state ve backend request payload temizlendi, join tarafı olduğu gibi bırakıldı.
- Room privacy sistemi eklendi: oda oluştururken `Herkese Açık` / `Özel` seçimi yapılabiliyor; private odalarda şifre alanı yalnız gerekli olduğunda gösteriliyor.
- Backend room sözleşmesine `room_type` ve `is_private` alanları eklendi; private odalar için şifre `bcrypt` ile hashlenerek `room_password_hash` olarak saklanıyor, API yanıtlarında expose edilmiyor.
- Join akışı iki aşamalı hale geldi: kullanıcı önce sadece oda kodu giriyor, oda private ise aynı form içinde ikinci adımda şifre isteniyor; public odalar mevcut akışla direkt katılıyor.
- Public room akışı, RoomPage timer/chat ve mevcut participant davranışı regresyon testleriyle doğrulandı.
- Theme initialization mantığı minimal şekilde güncellendi: localStorage'da tema tercihi varsa aynen kullanılıyor, kayıt yoksa uygulama artık sistem temasına bakmadan `light` ile açılıyor.
- Onboarding progress bölümünde alt segment/çizgili gösterge DOM’dan tamamen kaldırıldı; sadece ana progress bar ve `Adım x / 5` metni bırakıldı.
- Ana progress bar minimal polish ile hafif inceltildi; dark mode contrast korunarak mevcut dolum mantığı aynen bırakıldı.
- 5 adımlı yeni görünür onboarding akışı kuruldu: takma isim, handle, sınıf durumu, alan, çalışma sıklığı.
- Eski radio yapıları kaldırıldı; class status, study field ve study frequency için selectable card bileşenleri eklendi.
- Dark mode okunabilirliği güçlendirildi: başlık, açıklama, input, helper text, border, error state ve selected state kontrastları güncellendi.
- Handle girişi için daha net inline validation feedback eklendi; duplicate/invalid handle hatalarında kullanıcı yeniden handle adımına yönlendiriliyor.
- Çalışma sıklığı numeric input yerine preset seçeneklere taşındı ve mevcut backend beklentisine uygun `study_days` mapping’i ile gönderiliyor.
- Progress göstergesi sadeleştirildi; `Adım x / 5` ve modern progress bar eklendi.
- Test kapsamı için `backend/tests/test_onboarding_program_create_flow.py` regresyon testi eklendi.

## Prioritized Backlog
### P0
- Yok. Bu kapsam için onboarding akışı çalışır ve testlerden geçti.

### P1
- Backend genelinde bazı eski validation branch’lerini 200 + `{error: ...}` yerine tutarlı 4xx `HTTPException` yapısına taşımak.
- İleride sınıf durumu verisi ürün içinde kullanılacaksa mevcut sözleşmeyi bozmadan uygun veri modeli tasarlamak.

### P2
- Onboarding sonrası dashboard’da kullanıcı seçimlerinden türetilen daha kişisel ilk içerik/metinler sunmak.
- Onboarding analytics/event takibi eklemek.

## Next Tasks
- Kullanıcı isterse sınıf durumu bilgisini mevcut mimariye uygun ve kontrollü biçimde kalıcı hale getirmek.
- Gerekirse onboarding copy metinleri üzerinde küçük polish turu yapmak.
