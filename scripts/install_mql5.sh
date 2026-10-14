#!/usr/bin/env bash
# Copiaza scripturile MQL5 in folderul de date al MetaTrader 5.
#
# Ruleaza-l in Terminalul TAU (nu prin Claude):
#     bash scripts/install_mql5.sh
#
# Pe macOS, MT5 ruleaza sub Wine si isi tine datele intr-un bottle ascuns in ~/Library.
# Calea difera intre versiuni, asa ca scriptul o cauta in loc sa o presupuna.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$HERE/mql5"

if [ ! -d "$SRC" ]; then
  echo "Nu gasesc $SRC"; exit 1
fi

echo "Caut folderul de date MetaTrader 5..."
CANDIDATES=$(find "$HOME/Library/Application Support" \
                  -type d -name "Scripts" -path "*MQL5*" 2>/dev/null)

if [ -z "$CANDIDATES" ]; then
  cat <<'EOF'

Nu am gasit niciun folder MQL5/Scripts.

Cauze uzuale:
  - MetaTrader 5 nu a fost pornit niciodata (bottle-ul Wine se creeaza la prima pornire)
  - nu esti logat in niciun cont

Fa asa:
  1. Deschide MetaTrader 5 si logheaza-te in contul demo FTMO
  2. In MetaTrader:  File -> Open Data Folder
     (se deschide Finder exact unde trebuie)
  3. Intra in MQL5/Scripts si copiaza acolo, manual, cele doua fisiere din:
EOF
  echo "     $SRC"
  exit 1
fi

N=0
while IFS= read -r DIR; do
  [ -z "$DIR" ] && continue
  cp "$SRC"/ListSymbols.mq5 "$SRC"/ExportM1.mq5 "$DIR"/ 2>/dev/null && {
    echo "  copiat in: $DIR"
    N=$((N+1))
  }
done <<< "$CANDIDATES"

if [ "$N" -eq 0 ]; then
  echo "Am gasit foldere dar nu am putut scrie in ele. Copiaza manual din $SRC"
  exit 1
fi

cat <<'EOF'

Gata. Mai departe, in MetaTrader 5:

  1. Deschide MetaEditor:  Tools -> MetaQuotes Language Editor   (sau F4)
  2. In panoul Navigator din stanga: Scripts -> ListSymbols
     Dublu-click ca sa-l deschizi, apoi F7 ca sa compilezi.
     Trebuie sa scrie "0 errors, 0 warnings".
  3. Repeta pentru ExportM1.
  4. Inapoi in MetaTrader, in Navigator (Ctrl+N) -> Scripts.
     Daca nu apar, click dreapta -> Refresh.
  5. Trage ListSymbols pe ORICE grafic. Bifeaza "Allow modification of Signals
     settings" daca ti-o cere, apoi OK.
  6. Rezultatul: File -> Open Data Folder -> MQL5/Files/ftmo_symbols.csv

EOF
