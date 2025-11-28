# Generiranje sintetičkih vijesti pomocu LLM-ova

## Opis projekta

Ovaj projekt istražuje **generiranje sintetičkih (fake) vijesti** pomoću **velikih jezičnih modela (LLM-ova)** koristeći stvarne datasetove vijesti. Cilj je razviti **reproducibilan pipeline** koji uzima stvarne vijesti i generira stilizirane, namjerno izmijenjene članke. Projekt je edukativnog karaktera i naglašava metodologiju prompt engineeringa, preprocesiranje podataka i sigurnu generaciju sadržaja.  

Primjenom ovog projekta moguće je:
- Razviti razumijevanje rada LLM-ova u kontekstu generiranja teksta
- Analizirati razlike između stvarnih i sintetičkih vijesti
- Eksperimentirati s različitim promptovima i parametrima modela

---

## Dataset

Primarni dataset koristi **BBC News CSV** dostupan na Kaggleu:  
[https://www.kaggle.com/datasets/gpreda/bbc-news](https://www.kaggle.com/datasets/gpreda/bbc-news)  

Sadrži sljedeća polja (struktura CSV datoteke):
| Polje       | Opis                         |
|------------|------------------------------|
| `title`    | Naslov vijesti               |
| `pubDate`  | Datum objave                |
| `guid`     | Jedinstveni identifikator   |
| `link`     | URL članka                  |
| `description` | Kratki sadržaj vijesti    |

Dataset se može koristiti kao ulaz za generiranje sintetičkih vijesti.  

---

## Ciljevi projekta

1. **Prikupljanje i preprocesiranje podataka**
   - Čišćenje CSV/RSS podataka
   - Normalizacija tekstualnih polja (`title`, `description`)
   - Tokenizacija i segmentacija teksta prema potrebi

2. **Prompt engineering**
   - Razvijanje promptova za LLM za generiranje sintetičkih vijesti
   - Testiranje različitih stilova, tonova i duljina

3. **Generacija sintetičkih vijesti**
   - Korištenje LLM-a lokalno ili putem API-ja (npr. OpenAI, HuggingFace, Ollama)
   - Kontrola parametara generacije (temperature, max tokens, top_p)

4. **Evaluacija i analiza**
   - Usporedba sintetičkih i stvarnih vijesti
   - Detekcija sličnosti i stilskih promjena
   - Mogućnost vizualizacije distribucije tema i stilova

5. **Reproducibilan pipeline**
   - Google Colab notebook za jednostavno izvođenje
   - Jasan workflow od input CSV do generiranih sintetičkih članaka
   - Dokumentirani koraci s primjerima koda

---

## Tehnologije

- **Python 3.13**
- **Pandas / NumPy** – preprocesiranje i analiza podataka
- **Google Colab** – razvojno okruženje
- **LLM API ili lokalni LLM** – generacija sintetičkog sadržaja
- **Markdown / Jupyter Notebooks** – dokumentacija i vizualizacija workflowa

---

## Generacija sintetičke vijesti (pseudo-kod)

'''python
from llm_api import generate_text

prompt = f"Na temelju ovog članka generiraj stilizirani, izmijenjeni tekst: {df['description'][0]}"
synthetic_article = generate_text(prompt, temperature=0.7, max_tokens=300)
print(synthetic_article)

---

## Sigurnosna i etička razmatranja

Projekt nije namijenjen za širenje dezinformacija
Sintetičke vijesti su stilizirane i jasno označene kao generirane
Promovira istraživački i edukativni pristup LLM-ovima

---

## Prompt engineering i generacija sintetičkih vijesti

Primjer prompta:

Na temelju sljedećeg članka generiraj sintetički tekst:
{description}
Uvijek zadrži ton informativnosti, ali izmijeni detalje i stil.

---

## Primjeri rezultata

Stvarni članak:
"The government has announced new policies to reduce carbon emissions by 2030"

Sintetički članak (generiran LLM-om):
"Officials introduced fresh initiatives aimed at lowering greenhouse gases, with a target set for 2030. The new policies emphasize sustainable energy and urban planning"

---

## Autor

Lana Danolić