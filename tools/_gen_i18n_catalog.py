"""One-shot generator for modules/i18n/catalog.py — run from repo root."""
from __future__ import annotations

from pathlib import Path

# msgid, then es, pt-BR, fr, de, id, tr, it, pl, vi, nl
ROWS: list[tuple[str, ...]] = [
    ("Download", "Descargar", "Baixar", "Télécharger", "Download", "Unduh", "İndir", "Scarica", "Pobierz", "Tải xuống", "Downloaden"),
    ("Basic Settings", "Ajustes básicos", "Configurações básicas", "Paramètres de base", "Grundeinstellungen", "Pengaturan dasar", "Temel ayarlar", "Impostazioni di base", "Ustawienia podstawowe", "Cài đặt cơ bản", "Basisinstellingen"),
    ("Transcript && Subtitles", "Transcripción y subtítulos", "Transcrição e legendas", "Transcription et sous-titres", "Transkript && Untertitel", "Transkrip && Subtitle", "Transkript && Altyazılar", "Trascrizione e sottotitoli", "Transkrypcja i napisy", "Bản ghi && Phụ đề", "Transcript && ondertitels"),
    ("Advanced", "Avanzado", "Avançado", "Avancé", "Erweitert", "Lanjutan", "Gelişmiş", "Avanzate", "Zaawansowane", "Nâng cao", "Geavanceerd"),
    ("LLM Chat", "Chat LLM", "Chat LLM", "Chat LLM", "LLM-Chat", "Obrolan LLM", "LLM Sohbet", "Chat LLM", "Czat LLM", "Trò chuyện LLM", "LLM-chat"),
    ("Narrate", "Narración", "Narração", "Narration", "Erzählen", "Narasi", "Anlatım", "Narrazione", "Narracja", "Thuyết minh", "Vertellen"),
    ("Train", "Entrenar", "Treinar", "Entraîner", "Trainieren", "Latih", "Eğit", "Addestra", "Trenuj", "Huấn luyện", "Trainen"),
    ("Avoid", "Evitar", "Evitar", "Éviter", "Meiden", "Hindari", "Kaçın", "Evita", "Unikaj", "Tránh", "Vermijden"),
    ("About", "Acerca de", "Sobre", "À propos", "Info", "Tentang", "Hakkında", "Informazioni", "O programie", "Giới thiệu", "Over"),
    ("Interface language", "Idioma de la interfaz", "Idioma da interface", "Langue de l'interface", "Oberflächensprache", "Bahasa antarmuka", "Arayüz dili", "Lingua dell'interfaccia", "Język interfejsu", "Ngôn ngữ giao diện", "Interfacetaal"),
    ("Interface", "Interfaz", "Interface", "Interface", "Oberfläche", "Antarmuka", "Arayüz", "Interfaccia", "Interfejs", "Giao diện", "Interface"),
    ("Language", "Idioma", "Idioma", "Langue", "Sprache", "Bahasa", "Dil", "Lingua", "Język", "Ngôn ngữ", "Taal"),
    (
        "Choose the language for menus, tabs, and labels. Whisper and subtitle languages stay under Transcript settings.",
        "Elige el idioma de menús, pestañas y etiquetas. Los idiomas de Whisper y subtítulos siguen en Ajustes de transcripción.",
        "Escolha o idioma de menus, abas e rótulos. Os idiomas do Whisper e das legendas ficam em Configurações de transcrição.",
        "Choisissez la langue des menus, onglets et libellés. Les langues Whisper et sous-titres restent dans Transcription.",
        "Sprache für Menüs, Registerkarten und Beschriftungen. Whisper- und Untertitelsprachen bleiben unter Transkript.",
        "Pilih bahasa untuk menu, tab, dan label. Bahasa Whisper dan subtitle tetap di pengaturan Transkrip.",
        "Menüler, sekmeler ve etiketler için dili seçin. Whisper ve altyazı dilleri Transkript ayarlarında kalır.",
        "Scegli la lingua di menu, schede ed etichette. Le lingue Whisper e sottotitoli restano in Trascrizione.",
        "Wybierz język menu, kart i etykiet. Języki Whisper i napisów zostają w ustawieniach transkrypcji.",
        "Chọn ngôn ngữ cho menu, tab và nhãn. Ngôn ngữ Whisper và phụ đề vẫn ở phần Bản ghi.",
        "Kies de taal voor menu's, tabbladen en labels. Whisper- en ondertiteltalen blijven bij Transcript.",
    ),
    (
        "Language changes apply after you restart the app.",
        "Los cambios de idioma se aplican al reiniciar la aplicación.",
        "As alterações de idioma valem depois de reiniciar o aplicativo.",
        "Les changements de langue s'appliquent après redémarrage.",
        "Sprachänderungen gelten nach einem Neustart der App.",
        "Perubahan bahasa berlaku setelah Anda memulai ulang aplikasi.",
        "Dil değişiklikleri uygulamayı yeniden başlattıktan sonra geçerli olur.",
        "Le modifiche alla lingua si applicano dopo il riavvio dell'app.",
        "Zmiana języka obowiązuje po ponownym uruchomieniu aplikacji.",
        "Thay đổi ngôn ngữ có hiệu lực sau khi khởi động lại ứng dụng.",
        "Taalwijzigingen gelden nadat je de app opnieuw start.",
    ),
    ("Restart required", "Reinicio necesario", "Reinício necessário", "Redémarrage requis", "Neustart erforderlich", "Perlu mulai ulang", "Yeniden başlatma gerekli", "Riavvio necessario", "Wymagany restart", "Cần khởi động lại", "Herstart vereist"),
    (
        "The interface language will change after you close and reopen VideoHighlighter.",
        "El idioma de la interfaz cambiará al cerrar y volver a abrir VideoHighlighter.",
        "O idioma da interface mudará ao fechar e abrir o VideoHighlighter de novo.",
        "La langue de l'interface changera après fermeture et réouverture de VideoHighlighter.",
        "Die Oberflächensprache ändert sich, wenn Sie VideoHighlighter schließen und erneut öffnen.",
        "Bahasa antarmuka berubah setelah Anda menutup dan membuka lagi VideoHighlighter.",
        "Arayüz dili, VideoHighlighter'ı kapatıp yeniden açtığınızda değişir.",
        "La lingua dell'interfaccia cambierà dopo aver chiuso e riaperto VideoHighlighter.",
        "Język interfejsu zmieni się po zamknięciu i ponownym otwarciu VideoHighlighter.",
        "Ngôn ngữ giao diện sẽ đổi sau khi bạn đóng và mở lại VideoHighlighter.",
        "De interfacetaal verandert nadat je VideoHighlighter sluit en opnieuw opent.",
    ),
    ("OK", "Aceptar", "OK", "OK", "OK", "OK", "Tamam", "OK", "OK", "OK", "OK"),
    ("Download Videos from Website", "Descargar vídeos de un sitio web", "Baixar vídeos de um site", "Télécharger des vidéos depuis un site", "Videos von einer Website herunterladen", "Unduh video dari situs web", "Web sitesinden video indir", "Scarica video da un sito", "Pobierz filmy ze strony", "Tải video từ trang web", "Video's downloaden van een website"),
    ("Page URL:", "URL de la página:", "URL da página:", "URL de la page :", "Seiten-URL:", "URL halaman:", "Sayfa URL'si:", "URL della pagina:", "URL strony:", "URL trang:", "Pagina-URL:"),
    ("Save directory:", "Carpeta de destino:", "Pasta de destino:", "Dossier de destination :", "Zielordner:", "Direktori simpan:", "Kayıt klasörü:", "Cartella di salvataggio:", "Folder docelowy:", "Thư mục lưu:", "Map voor opslaan:"),
    ("Browse...", "Examinar...", "Procurar...", "Parcourir…", "Durchsuchen…", "Jelajahi...", "Gözat...", "Sfoglia…", "Przeglądaj…", "Duyệt...", "Bladeren…"),
    ("Download Time Range", "Rango temporal de descarga", "Intervalo de tempo do download", "Plage horaire du téléchargement", "Download-Zeitbereich", "Rentang waktu unduhan", "İndirme zaman aralığı", "Intervallo temporale download", "Zakres czasu pobierania", "Khoảng thời gian tải", "Downloadtijdsbereik"),
    ("Download:", "Descargar:", "Baixar:", "Télécharger :", "Download:", "Unduh:", "İndir:", "Scarica:", "Pobierz:", "Tải xuống:", "Downloaden:"),
    ("Full video", "Vídeo completo", "Vídeo completo", "Vidéo entière", "Ganzes Video", "Video penuh", "Tam video", "Video intero", "Cały film", "Toàn bộ video", "Volledige video"),
    ("Same range as processing", "Mismo rango que el procesamiento", "Mesmo intervalo do processamento", "Même plage que le traitement", "Gleicher Bereich wie Verarbeitung", "Rentang sama dengan pemrosesan", "İşleme ile aynı aralık", "Stesso intervallo dell'elaborazione", "Ten sam zakres co przetwarzanie", "Cùng khoảng với xử lý", "Zelfde bereik als verwerking"),
    ("Specific range (seconds)", "Rango concreto (segundos)", "Intervalo específico (segundos)", "Plage précise (secondes)", "Fester Bereich (Sekunden)", "Rentang tertentu (detik)", "Belirli aralık (saniye)", "Intervallo specifico (secondi)", "Konkretny zakres (sekundy)", "Khoảng cụ thể (giây)", "Specifiek bereik (seconden)"),
    ("Start time (seconds):", "Inicio (segundos):", "Início (segundos):", "Début (secondes) :", "Startzeit (Sekunden):", "Waktu mulai (detik):", "Başlangıç (saniye):", "Inizio (secondi):", "Czas początkowy (sekundy):", "Thời điểm bắt đầu (giây):", "Starttijd (seconden):"),
    ("End time (seconds):", "Fin (segundos):", "Fim (segundos):", "Fin (secondes) :", "Endzeit (Sekunden):", "Waktu selesai (detik):", "Bitiş (saniye):", "Fine (secondi):", "Czas końcowy (sekundy):", "Thời điểm kết thúc (giây):", "Eindtijd (seconden):"),
    ("Automatically add downloaded videos to file list", "Añadir automáticamente los vídeos descargados a la lista", "Adicionar automaticamente os vídeos baixados à lista", "Ajouter automatiquement les vidéos téléchargées à la liste", "Heruntergeladene Videos automatisch zur Dateiliste hinzufügen", "Otomatis tambahkan video terunduh ke daftar file", "İndirilen videoları otomatik olarak dosya listesine ekle", "Aggiungi automaticamente i video scaricati all'elenco", "Automatycznie dodawaj pobrane filmy do listy", "Tự động thêm video đã tải vào danh sách tệp", "Gedownloade video's automatisch aan de bestandenlijst toevoegen"),
    ("After download:", "Tras la descarga:", "Após o download:", "Après téléchargement :", "Nach dem Download:", "Setelah unduh:", "İndirmeden sonra:", "Dopo il download:", "Po pobraniu:", "Sau khi tải:", "Na download:"),
    ("Don't process", "No procesar", "Não processar", "Ne pas traiter", "Nicht verarbeiten", "Jangan proses", "İşlem yapma", "Non elaborare", "Nie przetwarzaj", "Không xử lý", "Niet verwerken"),
    ("Process each video as it downloads", "Procesar cada vídeo al descargarse", "Processar cada vídeo ao baixar", "Traiter chaque vidéo pendant le téléchargement", "Jedes Video bei Download verarbeiten", "Proses tiap video saat diunduh", "Her video indirdikçe işle", "Elabora ogni video durante il download", "Przetwarzaj każdy film w trakcie pobierania", "Xử lý từng video khi tải", "Elke video verwerken tijdens download"),
    ("Process all after downloads finish", "Procesar todo al terminar las descargas", "Processar tudo após os downloads", "Tout traiter après les téléchargements", "Alles verarbeiten, wenn Downloads fertig sind", "Proses semua setelah unduhan selesai", "İndirmeler bitince tümünü işle", "Elabora tutto a download terminati", "Przetwórz wszystko po zakończeniu pobierania", "Xử lý tất cả sau khi tải xong", "Alles verwerken als downloads klaar zijn"),
    ("Concurrent downloads:", "Descargas simultáneas:", "Downloads simultâneos:", "Téléchargements simultanés :", "Parallele Downloads:", "Unduhan bersamaan:", "Eşzamanlı indirme:", "Download simultanei:", "Równoległe pobierania:", "Tải đồng thời:", "Gelijktijdige downloads:"),
    ("Pick Videos from Page…", "Elegir vídeos de la página…", "Escolher vídeos da página…", "Choisir des vidéos sur la page…", "Videos von der Seite wählen…", "Pilih video dari halaman…", "Sayfadan video seç…", "Scegli video dalla pagina…", "Wybierz filmy ze strony…", "Chọn video từ trang…", "Video's van de pagina kiezen…"),
    ("Download All", "Descargar todo", "Baixar tudo", "Tout télécharger", "Alles herunterladen", "Unduh semua", "Tümünü indir", "Scarica tutto", "Pobierz wszystko", "Tải tất cả", "Alles downloaden"),
    ("Scoring Points", "Puntos de puntuación", "Pontos de pontuação", "Points de score", "Wertungspunkte", "Poin skor", "Puanlama", "Punti di punteggio", "Punkty oceniania", "Điểm số", "Scoringspunten"),
    ("Auto-Segmentation Settings", "Ajustes de auto-segmentación", "Configurações de auto-segmentação", "Paramètres d'auto-segmentation", "Auto-Segmentierung", "Pengaturan auto-segmentasi", "Otomatik segmentasyon ayarları", "Impostazioni auto-segmentazione", "Ustawienia auto-segmentacji", "Cài đặt tự phân đoạn", "Instellingen voor auto-segmentatie"),
    ("Transcript Settings", "Ajustes de transcripción", "Configurações de transcrição", "Paramètres de transcription", "Transkript-Einstellungen", "Pengaturan transkrip", "Transkript ayarları", "Impostazioni trascrizione", "Ustawienia transkrypcji", "Cài đặt bản ghi", "Transcriptinstellingen"),
    ("Source language:", "Idioma de origen:", "Idioma de origem:", "Langue source :", "Quellsprache:", "Bahasa sumber:", "Kaynak dil:", "Lingua di origine:", "Język źródłowy:", "Ngôn ngữ nguồn:", "Brontaal:"),
    ("Subtitle Settings", "Ajustes de subtítulos", "Configurações de legendas", "Paramètres des sous-titres", "Untertitel-Einstellungen", "Pengaturan subtitle", "Altyazı ayarları", "Impostazioni sottotitoli", "Ustawienia napisów", "Cài đặt phụ đề", "Ondertitelinstellingen"),
    ("Target language:", "Idioma de destino:", "Idioma de destino:", "Langue cible :", "Zielsprache:", "Bahasa target:", "Hedef dil:", "Lingua di destinazione:", "Język docelowy:", "Ngôn ngữ đích:", "Doeltaal:"),
    ("Avoid People", "Evitar personas", "Evitar pessoas", "Éviter des personnes", "Personen meiden", "Hindari orang", "İnsanlardan kaçın", "Evita persone", "Unikaj osób", "Tránh người", "Mensen vermijden"),
    ("Enable face recognition", "Activar reconocimiento facial", "Ativar reconhecimento facial", "Activer la reconnaissance faciale", "Gesichtserkennung aktivieren", "Aktifkan pengenalan wajah", "Yüz tanımayı etkinleştir", "Abilita riconoscimento facciale", "Włącz rozpoznawanie twarzy", "Bật nhận dạng khuôn mặt", "Gezichtsherkenning inschakelen"),
    ("When found:", "Si se encuentran:", "Quando encontradas:", "Si trouvées :", "Wenn gefunden:", "Jika ditemukan:", "Bulununca:", "Se trovate:", "Gdy znalezione:", "Khi tìm thấy:", "Indien gevonden:"),
    ("Skip those moments", "Omitir esos momentos", "Pular esses momentos", "Ignorer ces moments", "Diese Momente überspringen", "Lewati momen tersebut", "O anları atla", "Salta quei momenti", "Pomiń te momenty", "Bỏ qua những khoảnh khắc đó", "Die momenten overslaan"),
    ("Crop them out (experimental)", "Recortarlos (experimental)", "Cortar fora (experimental)", "Les recadrer (expérimental)", "Herausschneiden (experimentell)", "Potong keluar (eksperimental)", "Kırpıp çıkar (deneysel)", "Ritagliali (sperimentale)", "Wykadruj (eksperymentalne)", "Cắt bỏ (thử nghiệm)", "Uitsnijden (experimenteel)"),
    ("Updates", "Actualizaciones", "Atualizações", "Mises à jour", "Updates", "Pembaruan", "Güncellemeler", "Aggiornamenti", "Aktualizacje", "Cập nhật", "Updates"),
    ("Check for a newer version automatically", "Buscar automáticamente una versión más reciente", "Verificar automaticamente se há versão mais nova", "Rechercher automatiquement une version plus récente", "Automatisch nach neuerer Version suchen", "Periksa versi lebih baru secara otomatis", "Daha yeni sürümü otomatik denetle", "Cerca automaticamente una versione più recente", "Automatycznie sprawdzaj nowszą wersję", "Tự động kiểm tra phiên bản mới hơn", "Automatisch controleren op een nieuwere versie"),
    ("Check now", "Comprobar ahora", "Verificar agora", "Vérifier maintenant", "Jetzt prüfen", "Periksa sekarang", "Şimdi denetle", "Controlla ora", "Sprawdź teraz", "Kiểm tra ngay", "Nu controleren"),
    ("License", "Licencia", "Licença", "Licence", "Lizenz", "Lisensi", "Lisans", "Licenza", "Licencja", "Giấy phép", "Licentie"),
    ("Manage license…", "Administrar licencia…", "Gerenciar licença…", "Gérer la licence…", "Lizenz verwalten…", "Kelola lisensi…", "Lisansı yönet…", "Gestisci licenza…", "Zarządzaj licencją…", "Quản lý giấy phép…", "Licentie beheren…"),
    ("Copy", "Copiar", "Copiar", "Copier", "Kopieren", "Salin", "Kopyala", "Copia", "Kopiuj", "Sao chép", "Kopiëren"),
    ("Contact & Support", "Contacto y soporte", "Contato e suporte", "Contact et assistance", "Kontakt & Support", "Kontak & dukungan", "İletişim ve destek", "Contatti e supporto", "Kontakt i wsparcie", "Liên hệ & hỗ trợ", "Contact & ondersteuning"),
    (
        "Need help, found a bug, or have a feature request? Reach us here:",
        "¿Necesitas ayuda, encontraste un error o tienes una idea? Escríbenos aquí:",
        "Precisa de ajuda, achou um bug ou tem uma sugestão? Fale conosco:",
        "Besoin d'aide, un bug ou une idée ? Contactez-nous ici :",
        "Hilfe, Fehler oder Wunsch? Hier erreichen Sie uns:",
        "Butuh bantuan, menemukan bug, atau punya permintaan fitur? Hubungi kami di sini:",
        "Yardım, hata veya özellik isteği mi? Bize buradan ulaşın:",
        "Serve aiuto, hai trovato un bug o hai una richiesta? Scrivici qui:",
        "Potrzebujesz pomocy, znalazłeś błąd lub masz pomysł? Napisz tutaj:",
        "Cần trợ giúp, gặp lỗi hoặc muốn đề xuất? Liên hệ tại đây:",
        "Hulp nodig, een bug gevonden of een wens? Neem hier contact op:",
    ),
    ("Legal", "Legal", "Jurídico", "Mentions légales", "Rechtliches", "Legal", "Yasal", "Note legali", "Informacje prawne", "Pháp lý", "Juridisch"),
    ("View third-party notices…", "Ver avisos de terceros…", "Ver avisos de terceiros…", "Voir les avis tiers…", "Drittanbieter-Hinweise anzeigen…", "Lihat pemberitahuan pihak ketiga…", "Üçüncü taraf bildirimlerini gör…", "Visualizza avvisi di terze parti…", "Pokaż informacje o stronach trzecich…", "Xem thông báo bên thứ ba…", "Derde-partijmeldingen bekijken…"),
    ("Keep temp clips: ON", "Clips temporales: SÍ", "Cliques temporários: LIGADO", "Clips temporaires : OUI", "Temp-Clips: AN", "Klip sementara: AKTIF", "Geçici klipler: AÇIK", "Clip temporanei: ON", "Klipy tymczasowe: WŁ.", "Clip tạm: BẬT", "Tijdelijke clips: AAN"),
    ("Keep temp clips: OFF", "Clips temporales: NO", "Cliques temporários: DESLIGADO", "Clips temporaires : NON", "Temp-Clips: AUS", "Klip sementara: NONAKTIF", "Geçici klipler: KAPALI", "Clip temporanei: OFF", "Klipy tymczasowe: WYŁ.", "Clip tạm: TẮT", "Tijdelijke clips: UIT"),
    ("Export clips: ON", "Exportar clips: SÍ", "Exportar clipes: LIGADO", "Exporter les clips : OUI", "Clips exportieren: AN", "Ekspor klip: AKTIF", "Klipleri dışa aktar: AÇIK", "Esporta clip: ON", "Eksport klipów: WŁ.", "Xuất clip: BẬT", "Clips exporteren: AAN"),
    ("Export clips: OFF", "Exportar clips: NO", "Exportar clipes: DESLIGADO", "Exporter les clips : NON", "Clips exportieren: AUS", "Ekspor klip: NONAKTIF", "Klipleri dışa aktar: KAPALI", "Esporta clip: OFF", "Eksport klipów: WYŁ.", "Xuất clip: TẮT", "Clips exporteren: UIT"),
    ("Timeline Viewer", "Visor de línea de tiempo", "Visualizador da linha do tempo", "Visionneuse de timeline", "Timeline-Ansicht", "Penampil timeline", "Zaman çizelgesi", "Visualizzatore timeline", "Podgląd osi czasu", "Trình xem dòng thời gian", "Tijdlijnweergave"),
    ("Highlight Report", "Informe de highlights", "Relatório de highlights", "Rapport des highlights", "Highlight-Bericht", "Laporan highlight", "Öne çıkanlar raporu", "Report evidenziazioni", "Raport highlightów", "Báo cáo highlight", "Highlight-rapport"),
    ("AI Summary", "Resumen con IA", "Resumo com IA", "Résumé IA", "KI-Zusammenfassung", "Ringkasan AI", "Yapay zekâ özeti", "Riepilogo IA", "Podsumowanie AI", "Tóm tắt AI", "AI-samenvatting"),
    ("Log Output:", "Registro:", "Saída do log:", "Journal :", "Protokoll:", "Keluaran log:", "Günlük çıktısı:", "Output del log:", "Dziennik:", "Nhật ký:", "Loguitvoer:"),
    ("Drop a video here", "Suelta un vídeo aquí", "Solte um vídeo aqui", "Déposez une vidéo ici", "Video hier ablegen", "Letakkan video di sini", "Videoyu buraya bırakın", "Trascina qui un video", "Upuść film tutaj", "Thả video vào đây", "Zet hier een video neer"),
    (
        "or click to choose a file  ·  mp4, mov, mkv, avi",
        "o haz clic para elegir un archivo  ·  mp4, mov, mkv, avi",
        "ou clique para escolher um arquivo  ·  mp4, mov, mkv, avi",
        "ou cliquez pour choisir un fichier  ·  mp4, mov, mkv, avi",
        "oder klicken, um eine Datei zu wählen  ·  mp4, mov, mkv, avi",
        "atau klik untuk memilih file  ·  mp4, mov, mkv, avi",
        "veya dosya seçmek için tıklayın  ·  mp4, mov, mkv, avi",
        "oppure fai clic per scegliere un file  ·  mp4, mov, mkv, avi",
        "lub kliknij, aby wybrać plik  ·  mp4, mov, mkv, avi",
        "hoặc bấm để chọn tệp  ·  mp4, mov, mkv, avi",
        "of klik om een bestand te kiezen  ·  mp4, mov, mkv, avi",
    ),
    ("Analyze", "Analizar", "Analisar", "Analyser", "Analysieren", "Analisis", "Analiz et", "Analizza", "Analizuj", "Phân tích", "Analyseren"),
    ("Detailed settings", "Ajustes detallados", "Configurações detalhadas", "Paramètres détaillés", "Detaillierte Einstellungen", "Pengaturan terperinci", "Ayrıntılı ayarlar", "Impostazioni dettagliate", "Szczegółowe ustawienia", "Cài đặt chi tiết", "Gedetailleerde instellingen"),
    ("Detailed settings…", "Ajustes detallados…", "Configurações detalhadas…", "Paramètres détaillés…", "Detaillierte Einstellungen…", "Pengaturan terperinci…", "Ayrıntılı ayarlar…", "Impostazioni dettagliate…", "Szczegółowe ustawienia…", "Cài đặt chi tiết…", "Gedetailleerde instellingen…"),
    ("Simple view", "Vista simple", "Visão simples", "Vue simple", "Einfache Ansicht", "Tampilan sederhana", "Basit görünüm", "Vista semplice", "Widok prosty", "Chế độ đơn giản", "Eenvoudige weergave"),
    ("Run Highlighter", "Ejecutar Highlighter", "Executar Highlighter", "Lancer Highlighter", "Highlighter starten", "Jalankan Highlighter", "Highlighter'ı çalıştır", "Avvia Highlighter", "Uruchom Highlighter", "Chạy Highlighter", "Highlighter starten"),
    ("Cancel", "Cancelar", "Cancelar", "Annuler", "Abbrechen", "Batal", "İptal", "Annulla", "Anuluj", "Hủy", "Annuleren"),
    ("Add Videos", "Añadir vídeos", "Adicionar vídeos", "Ajouter des vidéos", "Videos hinzufügen", "Tambah video", "Video ekle", "Aggiungi video", "Dodaj filmy", "Thêm video", "Video's toevoegen"),
    ("Clear All", "Borrar todo", "Limpar tudo", "Tout effacer", "Alles löschen", "Hapus semua", "Tümünü temizle", "Cancella tutto", "Wyczyść wszystko", "Xóa tất cả", "Alles wissen"),
    ("Progress", "Progreso", "Progresso", "Progression", "Fortschritt", "Progres", "İlerleme", "Avanzamento", "Postęp", "Tiến độ", "Voortgang"),
    ("Processing Time Range", "Rango temporal de procesamiento", "Intervalo de tempo do processamento", "Plage horaire du traitement", "Verarbeitungszeitbereich", "Rentang waktu pemrosesan", "İşleme zaman aralığı", "Intervallo temporale elaborazione", "Zakres czasu przetwarzania", "Khoảng thời gian xử lý", "Verwerkingstijdsbereik"),
    ("Computer ID (for support):", "ID del equipo (soporte):", "ID do computador (suporte):", "ID ordinateur (assistance) :", "Computer-ID (Support):", "ID komputer (dukungan):", "Bilgisayar kimliği (destek):", "ID computer (supporto):", "ID komputera (wsparcie):", "ID máy (hỗ trợ):", "Computer-ID (ondersteuning):"),
    ("Pro license active.", "Licencia Pro activa.", "Licença Pro ativa.", "Licence Pro active.", "Pro-Lizenz aktiv.", "Lisensi Pro aktif.", "Pro lisans etkin.", "Licenza Pro attiva.", "Licencja Pro aktywna.", "Giấy phép Pro đang hoạt động.", "Pro-licentie actief."),
    (
        "Once a day, downloads a small text file listing the latest version. Nothing about you or this computer is sent.",
        "Una vez al día descarga un pequeño archivo de texto con la última versión. No se envía nada sobre ti ni este equipo.",
        "Uma vez por dia baixa um pequeno arquivo de texto com a versão mais recente. Nada sobre você ou este computador é enviado.",
        "Une fois par jour, télécharge un petit fichier listant la dernière version. Rien sur vous ni cet ordinateur n'est envoyé.",
        "Einmal täglich wird eine kleine Textdatei mit der neuesten Version geladen. Es werden keine Daten über Sie oder diesen Computer gesendet.",
        "Sekali sehari mengunduh file teks kecil berisi versi terbaru. Tidak ada data tentang Anda atau komputer ini yang dikirim.",
        "Günde bir, en son sürümü listeleyen küçük bir metin dosyası indirir. Sizinle veya bu bilgisayarla ilgili hiçbir şey gönderilmez.",
        "Una volta al giorno scarica un piccolo file di testo con l'ultima versione. Non viene inviato nulla su di te o su questo computer.",
        "Raz dziennie pobiera mały plik tekstowy z najnowszą wersją. Nic o tobie ani tym komputerze nie jest wysyłane.",
        "Mỗi ngày tải một tệp văn bản nhỏ liệt kê phiên bản mới nhất. Không gửi gì về bạn hoặc máy này.",
        "Eens per dag wordt een klein tekstbestand met de nieuwste versie gedownload. Er wordt niets over jou of deze computer verzonden.",
    ),
]

LANGS = ["es", "pt-BR", "fr", "de", "id", "tr", "it", "pl", "vi", "nl"]


def main() -> None:
    out: dict[str, dict[str, str]] = {code: {} for code in LANGS}
    for row in ROWS:
        assert len(row) == 1 + len(LANGS), row[0]
        msgid = row[0]
        for i, code in enumerate(LANGS):
            out[code][msgid] = row[1 + i]

    path = Path(__file__).resolve().parents[1] / "modules" / "i18n" / "catalog.py"
    lines = [
        '"""Phase-1 UI translations. Keys are English source strings (msgids)."""',
        "",
        "from __future__ import annotations",
        "",
        "# code -> {english_msgid: translation}",
        "TRANSLATIONS: dict[str, dict[str, str]] = {",
    ]
    for code in LANGS:
        lines.append(f"    {code!r}: {{")
        for k, v in out[code].items():
            lines.append(f"        {k!r}: {v!r},")
        lines.append("    },")
    lines.append("}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {path} ({len(ROWS)} strings x {len(LANGS)} languages)")


if __name__ == "__main__":
    main()
