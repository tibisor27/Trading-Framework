# Contractul Silver · `silver.bars` v1

Ce primește orice consumator (Gold, backtest, reconciliere) când citește Silver.
Producător: `data_platform/` (data engineering). Consumatori: tot ce citește `data/silver`.

Contractul are două jumătăți:
- **ce se poate verifica de cod** (forma și garanțiile) e în `contract.py`, obiectul `SILVER_BARS`. Producătorul îl rulează pe fiecare zi înainte s-o scrie, consumatorul după ce citește: `SILVER_BARS.validate(df)`. Textul de mai jos de la secțiunile 1 și 4 e generat cu `SILVER_BARS.describe()`.
- **ce înseamnă valorile** (sens, acoperire, prospețime) nu se poate verifica pe un singur fișier, deci e scris aici.

Orice schimbare de sens, nume, unitate sau oră înseamnă v2, anunț și re-măsurarea amprentei edge-ului.

## 1. Forma · executabil
```
silver.bars v1
  index   ts · datetime64[ns, UTC] · unique · increasing · on_minute
  open    float64 · finite and > 0
  high    float64 · finite and > 0
  low     float64 · finite and > 0
  close   float64 · finite and > 0
  volume  float64 · finite and >= 0
  check   high >= open, close
  check   low <= open, close
```
- Un fișier pe sursă, simbol și zi UTC: `data/silver/<source>/<symbol>/<YYYY-MM-DD>.parquet`, toate cele 24 de ore; toate barele în ziua fișierului.
- Lineage în metadatele fișierului (`pd.read_parquet(p).attrs`): sursa, fișierele Bronze, `ingested_at`, `built_at`.

## 2. Sensul
- `ts` = **începutul** minutului. Bara 09:30 acoperă 09:30:00–09:30:59, iar close-ul ei e cunoscut abia la 09:31 (de aici `act_offset_min: 1`). Verificat: randamentele Dukascopy și FTMO au corelația 0,999 cu ora aliniată și 0,04 decalate cu un minut.
- Prețul e cotația sursei, în puncte de indice. Dukascopy = BID. FTMO e în medie cu ~3 puncte (≈1 bp) peste Dukascopy. **Nu amesteca sursele în aceeași serie.**
- `volume` e în unitatea sursei: Dukascopy ≈ 0,08 pe minut, FTMO ≈ 386 de tick-uri pe minut. **Nu se compară între surse.**

## 3. Acoperirea
- **dukascopy**: 1.440 de bare pe zi. Când piața e închisă apar bare plate cu volum 0: pauza de seară
  16:15–18:00 New York la indicii SUA (20:15–22:00 UTC vara, 21:15–23:00 UTC iarna) și 17:00–18:00 New York
  la XAUUSD, weekendul (vineri seara → duminică 18:00 New York), sărbătorile, după 13:15 NY în zilele scurte.

- **mt5**: doar minutele cu tick-uri. Pauza de seară 20:50–22:05 UTC lipsește (~1.365 de bare pe zi).
- Fiecare zi văzută de Silver are o intrare în `_status.json`:
  - `ok`: fișierul există și a trecut verificările;
  - `empty`: sursa nu avea bare;
  - `rejected`: a încălcat contractul, `problems` spune de ce, iar fișierul nu există (sau e servită o versiune mai veche: `serving_previous`).

## 4. Garanțiile · executabil
Toate regulile din secțiunea 1, verificate de `silver.check` (= `SILVER_BARS.validate(df, day)`) înainte de scriere.
O zi care nu le respectă nu ajunge în Silver, iar motivul apare în `_status.json`. Nicio eroare nu se transmite consumatorilor.

## 5. Ce NU face Silver (sunt decizii pentru Gold)
Tăierea sesiunii · umplerea minutelor lipsă · filtrarea salturilor extreme · reeșantionarea (5m etc.) · feature-urile.

## 6. Prospețimea
- **dukascopy**: ziua D există după extragerea din D+1 și nu se mai schimbă.
- **mt5**: ziua D e definitivă după prima extragere făcută după sfârșitul ei. Până atunci poate fi parțială. *Încă nemarcat în status.*

## 7. Când ceva e greșit
Zilele afectate se reconstruiesc din Bronze (`python -m data_platform silver <inst> --all`). Nu se repară niciodată în Gold, în Bronze sau de mână în fișierele Silver.
