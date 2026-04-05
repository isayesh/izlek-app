# plan.md (Updated)

## 1) Objectives
- Dashboard ana sayfasını ve doğrudan kullandığı widget/component’leri **full-width, ferah, modern, premium SaaS** hissine taşımak.
- **Layout’u gerekirse baştan düzenlemek** (mevcut işleyen akışları/logic’i bozmadan).
- Hard rules:
  - **Yok:** `max-w-6xl/7xl`, `mx-auto`, tüm sayfayı tek boxed container’a alma.
  - **Var:** Full-width hissi, geniş ekran kullanımı, kontrol için `px-10/px-12` (responsive `px-4/px-6`).
- Header: sol üstte **text-based “izlek” brand** (logo yok; sonradan logo eklemek kolay), sağda mevcut navigation korunacak.
- Hero/top: “Merhaba, …” alanı büyütüldü + hızlı aksiyonlar eklendi.
- Kritik içerik grid’i:
  - İlk hedef 70/30’du; kullanıcı geri bildirimi ile premium denge için **yaklaşık 60/40** ayarlandı.
  - Desktop’ta: **sol (Bugün)** ≈ 60%, **sağ (Haftalık Program)** ≈ 40%.
- Haftalık program gün seçici:
  - Gün adları **tam** kalacak (kısaltma yok).
  - **Tek satır** görünüm (wrap yok): `Pazartesi → Pazar` aynı satırda.
- Kart dili (global): `rounded-2xl`, `p-6/p-7`, soft border, çok hafif shadow.
- Dark mode: saf siyah yok; **koyu lacivert tonlar**.

**Genişletilmiş uyumluluk hedefi (UI-only):**
- Dashboard ile aynı design language’i **dashboard-adjacent sayfalara** yaymak (yalnızca UI polish):
  - **Net Analizi (Net Takibi) sayfası**
  - **Online Çalışma Odaları (/rooms)**
  - **Oda içi sayfa (/room/:roomId)**

> Not: Tüm bu genişletmelerde kural: **Backend/logic/timer/realtime davranışına dokunma**, **layout/structure redesign yok**, sadece presentational/UI uyumu.

---

## Status (as of now)
- ✅ Latest GitHub `main` clone alındı ve /app ortamına sync edildi.
- ✅ Task UI-only kaldı, backend/API/logic’e dokunulmadı.
- ✅ Design guidelines üretildi ve uygulandı.
- ✅ Frontend bağımlılıkları `yarn install` ile düzeltildi (repo ile uyumlu).
- ✅ Preview/test için repo’nun mevcut dev auth bypass’ı kullanılacak şekilde `frontend/.env` içine `REACT_APP_FIREBASE_API_KEY=placeholder` eklendi (backend logic değişmeden).
- ✅ Stabil demo için backend’e seed data oluşturuldu ve `/app/memory/test_credentials.md` eklendi.

### Dashboard (Completed)
- ✅ Dashboard UI modernizasyonu uygulandı (full-width shell, yeni header/hero, quick actions, premium card dili, navy dark tokens).
- ✅ Hard-rule audit (dashboard scope): `max-w-6xl/7xl` ve `mx-auto` bulunmuyor.
- ✅ Testing Agent raporu: `/app/test_reports/iteration_1.json` — backend %100, frontend %100.
- ✅ Post-test: weekly tab/action + dialog butonları için ekstra `data-testid` eklendi; lint temiz.
- ✅ **Header brand polish:** sadece `izlek` text-brand güçlendirildi (dark mode subtle gradient). Test: `/app/test_reports/iteration_2.json`.
- ✅ **Weekly Program bug-fix:** `Pazar` görünürlüğü düzeltildi. Test: `/app/test_reports/iteration_3.json`.
- ✅ **Weekly Program refinement:** günler tek satır + oran rebalance `6fr/4fr`. Test: `/app/test_reports/iteration_4.json`.

### Net Analizi (Net Takibi) (Completed)
- ✅ Net Analizi sayfası dashboard tasarım dili ile uyumlu hale getirildi (layout/logic değişmeden).
- ✅ Test: `/app/test_reports/iteration_5.json` — frontend %100.

### Online Çalışma Odaları (/rooms) (Completed)
- ✅ /rooms sayfası dashboard design language ile uyumlu hale getirildi.
- ✅ Refine: Sayfa dashboard kopyası gibi hissettirmeyecek şekilde sadeleştirildi:
  - Profil özeti kaldırıldı
  - Aksiyon tekrarları kaldırıldı
  - Hero üstünde aksiyonlar, altta sadece seçili form görünüyor
  - Sağ panel denemesi yapıldı ve sonrasında kaldırıldı (yerine bir şey eklenmedi)
- ✅ Testler: `/app/test_reports/iteration_6.json` ve `/app/test_reports/iteration_7.json`.

### Oda içi sayfa (/room/:roomId) Visual Unification (Completed)
- ✅ Scope strict UI-only: logic/timer behavior/realtime akış/structure değişmedi.
- ✅ `/app/frontend/src/pages/RoomPage.js` dashboard design language’e hizalandı:
  - Purple ağırlıklı tema kaldırıldı → navy/soft dashboard paleti
  - Kartlar: rounded-2xl + soft shadow + surface tonu uyumu
  - Katılımcı listesi: daha sade, ağır border yok
  - Chat: daha okunur spacing + modern bubble yüzeyleri
  - Timer: daha görünür ama minimal (görsel)
  - Loading/error/not-found state’ler aynı görsel sistem
- ✅ Test: `/app/test_reports/iteration_9.json` — frontend %100, regresyon yok.

---

## 2) POC Decision
- **POC gerekmiyor** (mevcut çalışan app üzerinde yalnızca UI/UX ve layout refactor; yeni entegrasyon/back-end değişikliği yok).

---

## 3) Implementation Steps (Phases)

### Phase 1 — UI Baseline & Safety (No POC)
**User stories**
1. As a user, I want the dashboard to feel full-width so I can use my screen efficiently.
2. As a user, I want the header brand to look clean and premium even without a logo.
3. As a user, I want consistent spacing so I can scan sections comfortably.
4. As a user, I want cards to feel modern (soft borders, subtle shadows) so the UI feels premium.
5. As a user, I want dark mode to use navy tones so it’s comfortable and not harsh.

**Steps (Completed)**
- ✅ Repo’yu **latest `main`** üzerinden baz al; dashboard route ve doğrudan child component listesini çıkar.
- ✅ “Logic’e dokunmama” güvenliği:
  - State/handlers/data fetching fonksiyonlarına dokunmadan sadece layout/wrapper/markup className düzenleri.
  - Gerekirse sadece presentational split (aynı props, aynı davranış).
- ✅ Tasarım token’ları için minimal shared layer:
  - Global CSS token’ları ve dark mode yüzeyleri navy tonlara çekildi.
  - Shared UI (Card/Button/Tabs gibi) component’lerinde **stil standardizasyonu** uygulandı.

**Output (Completed)**
- ✅ Değişecek dosyaların aday listesi netleşti ve uygulandı (dashboard page + shared ui/style).

---

### Phase 2 — V1 Dashboard Layout (Core UI Revamp)
**User stories**
1. As a user, I want a larger greeting/hero area so the dashboard feels welcoming and not cramped.
2. As a user, I want quick actions right under the hero so I can act fast.
3. As a user, I want the main content and weekly plan in a balanced split so priorities are clear.
4. As a user, I want sections to have breathing room so the page doesn’t look like an admin panel.
5. As a user, I want the dashboard to remain fully functional (same data, same actions) after the redesign.

**Steps (Completed)**
- ✅ Page shell: full-width padding (`px-4 sm:px-6 lg:px-10 xl:px-12`), daha ferah vertical rhythm.
- ✅ Header redesign: text-based `izlek` brand + sağda nav aksiyonları.
- ✅ Hero/top: büyütülmüş selamlama + hızlı aksiyonlar.
- ✅ Main grid: kullanıcı geri bildirimine göre `lg:grid-cols-[minmax(0,6fr)_minmax(0,4fr)]`.
- ✅ Haftalık gün seçici: tam gün isimleri, tek satır.
- ✅ Dark mode: saf siyah yok; navy yüzeyler.

**Testing (Completed)**
- ✅ `/app/test_reports/iteration_1.json` (dashboard).
- ✅ `/app/test_reports/iteration_2.json` (header brand).
- ✅ `/app/test_reports/iteration_3.json` (Pazar fix).
- ✅ `/app/test_reports/iteration_4.json` (tek satır + oran).

---

### Phase 3 — Polish, Consistency & Hard-Rules Audit
**User stories**
1. As a user, I want consistent typography so headings and sections feel cohesive.
2. As a user, I want subtle hover/focus states so the UI feels high quality.
3. As a user, I want the weekly plan and today tasks to be visually balanced.
4. As a user, I want empty/loading states to look clean.
5. As a user, I want no boxed layout artifacts.

**Steps (Completed / Verified)**
- ✅ Hard rules denetimi (dashboard scope)
- ✅ Micro-interactions (buton/kart)
- ✅ Loading/empty state polish

---

### Phase 4 — Page Alignment: Net Analizi (Net Takibi) UI Polish
**User stories**
1. As a user, I want Net Analizi sayfasının dashboard ile aynı tasarım dilini konuşmasını.
2. As a user, I want inputs to feel modern and responsive with clear focus states.
3. As a user, I want empty states to feel soft and premium.
4. As a user, I want no layout/logic changes.

**Steps (Completed)**
- ✅ Kartlar: rounded-2xl, p-6, soft shadow
- ✅ Başlıklar: `text-xl`
- ✅ Inputs: soft background + improved focus ring
- ✅ Empty state: soft/centered
- ✅ Dark mode: dashboard ile uyum

**Testing (Completed)**
- ✅ `/app/test_reports/iteration_5.json`

---

### Phase 5 — Rooms: Online Çalışma Odaları (/rooms) UI Alignment + Refinement
**User stories**
1. As a user, I want rooms sayfasının aynı ürün ailesinden olduğunu hissetmek.
2. As a user, I want the page to be action-driven and not repetitive.
3. As a user, I want quick actions and the relevant form without duplicate UI.

**Steps (Completed)**
- ✅ Wide container/padding dashboard ile hizalandı.
- ✅ Hero: title + description + action buttons.
- ✅ Duplikasyon temizliği:
  - Profil summary kaldırıldı.
  - Alt taraftaki tekrar eden create/join chooser kaldırıldı.
  - Altta yalnızca seçili form gösteriliyor.
- ✅ Sağ panel denemesi kaldırıldı, hero temiz/balanced bırakıldı.
- ✅ Spacing/padding küçük denge ayarları yapıldı.

**Testing (Completed)**
- ✅ `/app/test_reports/iteration_6.json`
- ✅ `/app/test_reports/iteration_7.json`

---

### Phase 6 — Room Page (/room/:roomId) Visual Unification
**User stories**
1. As a user, I want the room experience to look like the dashboard’s sibling.
2. As a user, I want chat to be readable and modern.
3. As a user, I want timer to be clear without feeling “gamified”.
4. As a user, I want no real-time/timer behavior changes.

**Steps (Completed)**
- ✅ Colors: purple-heavy görünüm kaldırıldı, dashboard palette’ine alındı.
- ✅ Cards: rounded-2xl + soft shadow + surface tonu.
- ✅ Chat: spacing/readability polish; bubble yüzeyleri dashboard tone.
- ✅ Timer: daha görünür ama minimal (yalnızca stil).
- ✅ User list: border-heavy görünüm kaldırıldı, daha clean.
- ✅ Loading/error/not-found state’ler dashboard token’larıyla uyumlu.

**Testing (Completed)**
- ✅ `/app/test_reports/iteration_9.json` — frontend %100.

---

## 4) Next Actions
- ✅ (Done) Dashboard UI revamp + micro-iterations + test.
- ✅ (Done) Net Analizi UI polish + test.
- ✅ (Done) Rooms (/rooms) UI alignment + refinement + test.
- ✅ (Done) Room page (/room/:roomId) visual unification + test.
- ⏭️ (Delivery housekeeping) Değiştirilen dosyaları listele, kısa açıklama yaz, test raporlarını referansla.

---

## 5) Success Criteria
- ✅ Dashboard full-width, ferah, premium.
- ✅ Hard rules: dashboard scope’ta `max-w-6xl/7xl` ve `mx-auto` yok.
- ✅ Header brand: daha görünür, box/badge yok, dark mode subtle gradient.
- ✅ Weekly Program:
  - Günler eksiksiz (Pazartesi→Pazar)
  - Tek satır (wrap yok)
  - Oran dengesi: `6fr/4fr`
- ✅ Net Analizi: dashboard dilinde, UI-only polish.
- ✅ Rooms:
  - Dashboard dilinde ama dashboard kopyası değil
  - Duplikasyon yok, sadece seçili form var
- ✅ Room page:
  - Dashboard dilinde (navy/soft)
  - Chat/timer/user list modern ve clean
  - Timer/realtime davranışı değişmeden kaldı
- ✅ Testler: iteration 1–5, 6–7, 9 raporlarıyla doğrulandı.
