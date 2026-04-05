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
