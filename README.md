# Валютын зах зээлийн чиг хандлагыг таамаглах арилжааны систем — өгөгдлийн давхарга

1-р долоо хоног (11-р сар): орчин, EUR/USD 2015–2025 оны 1 цагийн лаа, мэдээний календарь, нэгтгэл, өгөгдлийн сан.

## Бүтэц
```
scripts/download_dukascopy.py   # үндсэн эх сурвалж (dukascopy-node@1.51.1, Node.js 20+)
scripts/download_histdata.py    # хөндлөн шалгах 2 дахь эх сурвалж
scripts/run_pipeline.py         # унших -> чанар -> хөндлөн шалгалт -> мэдээ нэгтгэх -> DB
src/fxdata/prices.py            # CSV/zip унших, UTC, M1->H1
src/fxdata/quality.py           # цоорхой, spike, OHLC зөрчил, давхардал, эх сурвалж харьцуулах
src/fxdata/calendar.py          # календарь, цагийн бүс, '175K'->175000, surprise, surprise_z
src/fxdata/merge.py             # мэдээ <-> лаа холбоос, look-ahead-гүй мэдээний шинжүүд
src/fxdata/timing.py            # мэдээний цаг зөв лаанд буусныг шалгах (event study)
sql/schema.sql                  # өгөгдлийн сангийн бүтэц
tests/test_pipeline.py          # 18 тест (синтетик өгөгдөл)
```

## Ажиллуулах (Windows/Mac/Linux)
```bash
python -m venv .venv && .venv\Scripts\activate      # Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q                                  # 18 passed байх ёстой

python scripts/download_dukascopy.py --from-year 2015 --to-year 2025
python scripts/download_histdata.py  --from-year 2015 --to-year 2025

# Календарь: data/raw/calendar/forex_factory.csv гэж хадгална.
# ЦАГИЙН БҮСИЙГ ЗААВАЛ ЗӨВ ЗААНА (HuggingFace-ийн ForexFactory датасет = Asia/Tehran).
python scripts/run_pipeline.py --dukascopy-dir data/raw/dukascopy --histdata-dir data/raw/histdata \
    --calendar data/raw/calendar/forex_factory.csv --calendar-tz Asia/Tehran
```
Гаралт: `data/fx.db`, `data/processed/eurusd_h1.parquet`, `eurusd_h1_with_news.parquet`,
`quality_issues.csv`, `pipeline_report.json` (илтгэлийн тоонууд эндээс гарна).

## Цагийн дүрэм
- Бүх цаг UTC. Лааны цаг = нээлтийн цаг (12:00 лаа = 12:00–13:00).
- 12:30-д гарсан мэдээ 12:00 лаанд орно. Түүний actual-ийг зөвхөн 13:00-аас хойш ашиглана.
- HistData: **America/New_York** (зуны цагтай). Баримт бичигт "EST without DST" гэж бичсэн ч
  бодит файлууд Dukascopy-тай зөвхөн ингэж таарна (өмнө нь тогтмол +5 цаг гэж үзээд зуны улиралд 1 цаг зөрж байсан).
- ForexFactory (HuggingFace) календарь:
  - Файлын UTC offset-д итгэхгүй. 2023 оноос хойш зуны улиралд +04:30 гэж буруу бичсэн
    (Иран 2023 оноос зуны цаг хэрэглэхээ больсон). Wall time-ийг tzdata-аар Asia/Tehran бүсээр хөрвүүлнэ.
  - USD/EUR өндөр нөлөөтэй мэдээний ~40–50% нь `00:00:00` (цаггүй). Цагийг `calendar.SCHEDULE`
    (албан ёсны хуваарь) -> тухайн оны давамгай цаг -> бүх жилийн давамгай цаг гэсэн дарааллаар сэргээнэ.
    Сэргээгдээгүй (илтгэл, уулзалт) мөрийг лаатай холбохгүй. `econ_events.time_source` баганаар эх сурвалжийг нь хянана.
  - Календарь 2025-04-07-нд дуусдаг тул `eurusd_h1_with_news.parquet` тэр хүртэл л үргэлжилнэ.
- surprise_z-ийг шинжид ±5-аар хязгаарлана (COVID үеийн z≈100). DB-д анхны утга хадгалагдана.

## Шалгах зүйлс (pipeline_report.json)
- `cross_check.close_mean_pips`: ойролцоогоор 1 пипээс бага байх ёстой (засварын дараа 0.41, өмнө нь 4.44 байсан).
  `shift±1h` нь 0-д ойр байвал цагийн бүсийн алдаа байна.
- `news_timing_check`: мэдээний лааны хүрээ / ердийн хүрээ харьцааны оргил `peak_shift = 0` дээр байх ёстой
  (`news_timing_check.png`). Засварын өмнө/дараах харьцуулалт: `news_timing_before_after.png`.
  ECB-ийн шийдвэрийн оргил +1 цагт гардаг нь хэвийн: зах зээл шийдвэрээс илүү 45 минутын дараах хэвлэлийн бага хуралд хүчтэй хариу үзүүлдэг.
- `spikes_on_high_news_hours / spikes_total`: том лааны ихэнх нь мэдээний цагт таарвал бодит хөдөлгөөн
  (засварын дараа 58/91, өмнө нь 37/91).
