# Prepoznavanje glazbenog žanra pomoću strojnog učenja

**Prateći dokument uz `genre_notebook_sci.ipynb`**

## Uvod

Ovaj dokument ide uz bilježnicu i objašnjava zašto je projekt strukturiran na ovaj način, koje su odluke donesene i koje su granice ovoga projekta. Bilježnica sadrži kod, grafove i rezultate. Ovdje pišem o razlozima iza koda, o propustima koje sam kasnije ispravila, te odgovaram na jedanaest pitanja o projektu.

Bilježnica se jednostavno pokrene sa Start/Run code. U njoj su već ispisani rezultati pokretanja.

## O projektu

Cilj projekta je provjeriti može li se glazbeni žanr pjesme predvidjeti isključivo iz numeričkih audio značajki (danceability, energy, tempo, valence, acousticness, itd.) koje dolaze iz Kaggle dataseta preuzetog putem `kagglehub` dataseta, bez teksta, slike omota ili metapodataka o izvođaču. Dataset sadrži 14 audio značajki i 11 žanrova (Acoustic/Folk, Alt_Music, Blues, Bollywood, Country, HipHop, Indie Alt, Instrumental, Metal, Pop, Rock). Zadatak je klasifikacija u više klasa, dodatno otežana time što su neki od tih žanrova zvučno vrlo bliski (npr. Alt_Music i Indie Alt).

Pipeline u bilježnici prolazi kroz sljedeće korake:

1. **Preuzimanje podataka** direktno s Kagglea (`kagglehub`
2. **Eksploratorna analiza** - pregled distribucije žanrova, nedostajućih vrijednosti i korelacija među značajkama
3. **Predobrada** - troslojna podjela na train/validation/test, popunjavanje nedostajućih vrijednosti medijanom i standardizacija, pri čemu se obje statistike računaju isključivo na train skupu
4. **Treniranje** - usporedba dva trivijalna baseline pristupa (uvijek predvidi najčešći žanr; nasumično po učestalosti) s Logistic Regression i Random Forest modelima
5. **Evaluacija** - classification report, matrica zabune, izdvojena analiza zašto se Alt_Music sustavno loše prepoznaje, te provjera je li prijavljena "pouzdanost" predviđanja stvarno kalibrirana
6. **Konačna provjera** na test skupu, koji se do tog trenutka nije dirao, kako bi izvještena točnost bila nepristrana procjena
7. **Spremanje modela** i funkcija za predviđanje žanra jedne pjesme
8. **Interaktivni upload** - korisnik može uploadati CSV s audio značajkama (precizno) ili pravu mp3/wav datoteku (eksperimentalno, jer se Spotify-ove značajke aproksimiraju iz sirovog zvuka pomoću librose)

Konačni model (Random Forest) postiže oko 51-52% točnosti na validaciji i sličnu točnost na test skupu, što je otprilike pet puta bolje od slučajnog pogađanja i skoro dvostruko bolje od naivnog baseline-a koji uvijek predviđa najčešći žanr. To je solidan, ali daleko od savršenog rezultat, i najveći dio ovog dokumenta bavi se time zašto je to tako i gdje su granice pouzdanosti tog broja.

Bitno je reći da projekt ima i eksperimentalni, "zabavniji" dio - mogućnost da se pravu pjesmu (mp3/wav) uploada i dobije predviđeni žanr, gdje se audio značajke procjenjuju librosa bibliotekom umjesto da dolaze izravno iz Spotify API-ja. Taj dio namjerno tretiram kao demonstraciju koncepta, ne kao provjerenu funkcionalnost - razlog je detaljno objašnjen u pitanju 7 niže.

## Pitanja i odgovori

U ovom dijelu odgovaram na pitanja o metodologiji i rezultatima projekta. Dio odgovora (5, 6, 2, 3, 8) sam nakon razmišljanja o njima i ugradila u bilježnicu, ne samo teoretski opisala. Uveden je odvojen test skup koji se dira samo jednom (poglavlje 6), medijani i skaliranje sada se računaju isključivo na train skupu (poglavlje 3), dodana je usporedba s dva trivijalna baseline pristupa (poglavlje 4), dodana je konkretna analiza zašto Alt_Music loše prolazi (poglavlje 5.1) i provjera kalibracije pouzdanosti (poglavlje 5.2). Ostatak odgovora (pitanja 7 i 10) ostaje opisan teoretski jer bi zahtijevao dodatne resurse (npr. skup označenih audio datoteka) koje trenutno nemam.

### 1. Koje je glavno istraživačko pitanje Vašeg projekta? Što ste konkretno željeli provjeriti i koji bi rezultat predstavljao uspjeh projekta?

Glavno pitanje je može li se glazbeni žanr pjesme predvidjeti isključivo iz numeričkih audio značajki (danceability, energy, tempo, valence, acousticness, itd.), bez ikakvog teksta, slike omota ili metapodataka o izvođaču. Konkretno sam htjela provjeriti nosi li ovakav skup značajki dovoljno informacija da predviđanje bude bolje od nasumičnog pogađanja i jednostavnih osnovnih modele. Također sam htjela provjeriti koliko se ti žanrovi međusobno razlikuju. Uspjeh projekta ne definiram kao "visoku točnost" u apsolutnom smislu, nego kao jasan i mjerljiv pomak u odnosu na osnovne pristupe, uz razumijevanje za koje je žanrove taj pristup pouzdan, a za koje nije.

### 2. Što znači rezultat od približno 50,7 posto točnosti? Je li to dobar ili loš rezultat za problem s jedanaest žanrova? S kojim jednostavnim početnim pristupom biste ga trebali usporediti prije donošenja zaključka?

Sam po sebi, taj broj ne govori puno dok ga se ne usporedi s nečim. Zato sam u poglavlju 4 dodala stvarno izračunatu usporedbu s dva baseline pristupa, umjesto da to ostane samo teoretska napomena: "uvijek predvidi najčešći žanr" i "nasumično po učestalosti". Kod 11 klasa, potpuno slučajno pogađanje dalo bi otprilike 9,1% točnosti (1/11), a baseline koji uvijek predviđa najčešći žanr u bazi (Rock) u praksi izlazi oko 27-28%. Model je dakle otprilike 1,8 puta bolji od "uvijek pogodi najčešći žanr" pristupa i više nego pet puta bolji od slučajnog pogađanja, što znači da nešto uči iz značajki. To je solidan rezultat za problem s 11 preklapajućih žanrova.

### 3. Model neke žanrove prepoznaje vrlo dobro, a Alt_Music gotovo uopće ne prepoznaje. Što nam to govori o skupu podataka, definiciji žanrova ili odabranim značajkama? Koje biste dodatne provjere napravili prije zaključka da je problem u modelu?

Ovo sam konkretno provjerila u poglavlju 5.1, umjesto da samo pretpostavim odgovor na temelju konteksta. Iz matrice zabune izdvojene za Alt_Music vidi se kamo model najčešće "šalje" te pjesme - u ovom skupu podataka to su najvećim dijelom Rock i Indie Alt. Grafovi distribucije značajki (energy, acousticness, valence) za Alt_Music naprema Indie Alt i Rock pokazuju znatno preklapanje, što je jak signal da problem nije samo u modelu, nego u tome da same audio značajke ne sadrže dovoljno informacije da razdvoje te žanrove - ili da granica između njih uopće nije dobro definirana u samom procesu označavanja podataka. Prije konačnog zaključka da je kriv model, dodatno bih provjerila: je li Alt_Music sustavno manja klasa u odnosu na slične žanrove, i kako je dataset uopće došao do tih oznaka žanra. 

### 4. Koliko su oznake glazbenih žanrova objektivne? Može li ista pjesma opravdano pripadati različitim žanrovima? Kako takva neodređenost utječe na treniranje i evaluaciju??

Ne bih rekla da su posve objektivne. Žanr nije fizičko svojstvo zvuka poput frekvencije ili trajanja, nego kulturna i marketinška kategorija koja se mijenja kroz vrijeme i često ovisi o kontekstu izvođača, generaciji, kulturi, te čak o tome tko je pjesmu kategorizirao za potrebe streaming platforme. Ista pjesma sasvim opravdano može istovremeno pripadati više žanrova - primjer je gotovo bilo koja pjesma na granici Pop/Alt_Music ili Country/Acoustic-Folk. Ovaj dataset prisiljava svaku pjesmu na točno jedan žanr, što miče tu nesigurnost i pretvara se u šum. Tijekom treniranja to znači da model uči na djelomično proturječnim primjerima, a tijekom evaluacije to znači da metrike kažnjavaju model i za predviđanja koja su u stvarnosti sasvim razumna. Zbog toga je gornja granica ostvarive točnosti realno niža od 100% bez obzira na kvalitetu modela.

### 5. Podaci se dijele na skup za treniranje i validaciju, ali se isti validacijski skup koristi i za odabir boljeg modela i za izvještavanje rezultata. Zašto bi rezultat zbog toga mogao izgledati boljim nego što stvarno jest? Kako biste organizirali pouzdaniju završnu provjeru?

Kada se dva modela isprobaju na istom validacijskom skupu i zatim odabere onaj s boljim rezultatom, taj postupak unosi pristranost prema tom skupu podataka - odabran je model koji je "slučajno" najbolje pogodio baš te primjere, što ne mora značiti da će jednako dobro raditi na drugim podacima. Zato je pouzdanije podatke podijeliti na tri dijela: skup za treniranje, skup za validaciju (isključivo za odabir modela) i potpuno odvojen testni skup koji se iskorištava samo jednom, na samom kraju. To sam napravila u poglavljima 3 i 6 u bilježnici - dodana je troslojna podjela, a test skup se koristi tek u posljednjoj ćeliji poglavlja 6, nakon što je model već odabran na validaciji. Razlika između validacijske i test točnosti (obje ispisane u poglavlju 6) pokazuje koliko je odabir modela bio "namješten" prema validacijskom skupu - ako je razlika mala, to je dobar znak.

### 6. Nedostajuće vrijednosti popunjavaju se medijanima izračunatim prije podjele podataka. Zašto bi obrada podataka trebala biti određena samo na skupu za treniranje? Objasnite problem vlastitim riječima, bez potrebe za pisanjem koda.

Ideja validacije je simulirati situaciju u kojoj model susreće potpuno nove, dosad neviđene podatke. Ako se medijan (ili bilo koja druga statistika korištena za predobradu) izračuna na cijelom skupu podataka prije podjele, onda validacijski primjeri neizravno utječu na tu vrijednost - model tako "zna" nešto o distribuciji validacijskih podataka i prije nego što ih formalno vidi. U prvoj verziji bilježnice medijan se računao na cijelom skupu prije podjele (`medians = X.median()` prije `train_test_split()`), što sam ispravila u poglavlju 3 - sada se `medians = X_train.median()` računa isključivo nakon podjele, na train skupu, i te iste vrijednosti se zatim primjenjuju i na validaciju i na test.

### 7. Audio datoteka obrađuje se pomoću značajki koje samo približno odgovaraju Spotify značajkama. Imamo li dovoljno dokaza da su takva predviđanja pouzdana? Osmislite eksperiment kojim biste provjerili rade li predviđanja iz stvarnih audio datoteka bolje od slučajnog pogađanja.

Ne u smislu formalnog eksperimenta s p-vrijednostima, ali sam napravila brzu provjeru. Uploadala sam nekoliko pjesama svojih omiljenih izvođača iz raznih žanrova i pogledala što model kaže. Isprobala sam Lovejoy, Bears in Trees, Pink Floyd, Quadeca, Marina and the Diamonds i Lemon Demon. Za Marina and the Diamonds, Bears in Trees i Pink Floyd predviđeni žanr mi je djelovao otprilike točno, gdje su glavni i podžanrovi često bili slični. Za Quadeca, Lemon Demon i Lovejoy model je često skakao po žanrovima i podžanrovima ovisno o pjesmi, te ponekad i krivo odredio iste. Kod eksperimentalnijih pjesama tih izvođača predviđanje ponekad nije imalo puno smisla. Sva tri izvođača svjesno mijenjaju zvuk iz pjesme u pjesmu, pa me to nije iznenadilo. 
To isto tako znači da ne možemo reći je li izvor nedosljednosti model koji ispravno hvata tu promjenu zvuka ili model koji je jednostavno zbunjen i otprilike pogađa po najbližim značajkama.

To mi daje neki kvalitativan osjećaj da pipeline barem djelomično radi razumno, ali daleko je od "dovoljno dokaza" u znanstvenom smislu. Šest izvođača i par pjesama je premalen i previše subjektivan uzorak - ja sam ta koja je procjenila jesu li predviđanja "otprilike točno". Tu nisu uključene formalne oznake, nema statističke značajnosti, i namjerno sam birala pjesme koje smatram da su eksperimentalnije za provjeru. Pravi eksperiment bi izgledao ovako: uzeti veći, unaprijed pripremljen skup pjesama s pouzdano poznatim žanrom (idealno iz nekog postojećeg većeg audio dataseta), pustiti sve kroz izvuci_znacajke_iz_audia pa predict_genre, izbrojati koliko je predviđanja pogodilo stvarni žanr, i tu točnost usporediti sa slučajnim pogađanjem (9,1% za 11 klasa) uz binomni test značajnosti. Ako razlika ne ispadne statistički značajna, to bi značilo da moj test nije reprezentativan.

### 8. Program prikazuje "pouzdanost" predviđanja. Što bi trebalo provjeriti prije nego što broj poput 38,8 posto korisniku predstavimo kao pouzdanost? Može li precizno prikazan broj ipak biti metodološki nepouzdan?

Treba provjeriti je li ta vjerojatnost kalibrirana - odnosno, od svih predviđanja kojima model dodijeli otprilike 38,8% vjerojatnosti, je li stvarno oko 38,8% njih točno. Taj reliability dijagram sam napravila u poglavlju 5.2 - iz njega je vidljivo koliko se prijavljena pouzdanost razlikuje od stvarne točnosti u pojedinim rasponima. Ako točke leže ispod dijagonale, model je preoptimističan, pa se pouzdanost trenutno ne bi trebala prikazivati korisniku kao strogo statistički broj bez te ograde.

Precizno prikazan broj itekako može biti metodološki nepouzdan. Random Forest te vjerojatnosti dobiva iz omjera stabala u šumi, što je matematički precizan broj, ali ne mora imati nikakve veze sa stvarnom vjerojatnošću točnog pogotka dok se to eksplicitno ne provjeri.

### 9. Koji su najvažniji zaključci projekta, osim činjenice da je jedan model ostvario veću točnost od drugoga? Navedite barem dva zaključka koja proizlaze iz podataka i rezultata te jedno važno ograničenje istraživanja.

Prvi zaključak je da audio značajke same po sebi dobro razdvajaju žanrove s prepoznatljivim, distinktivnim zvukom (Instrumental, Bollywood, Country, Acoustic/Folk su svi imali f1-mjeru iznad 0,7), što pokazuje da odabrane značajke nose stvaran, koristan signal, a ne samo šum. Drugi zaključak je da granice između sličnih podžanrova (Alt_Music/Indie Alt) audio značajke ne uspijevaju pouzdano povući (potvrđeno konkretnom analizom u poglavlju 5.1), što je vjerojatno kombinacija stvarnog zvučnog preklapanja i nedosljednosti u samom označavanju podataka, a ne samo slabost algoritma. Najvažnije ograničenje istraživanja je da su oznake žanra u skupu podataka preuzete kao "istina" bez ikakve provjere kako su nastale niti koliko su pouzdane i da cijela evaluacija implicitno pretpostavlja da su te oznake točne.

### 10. Kako ste koristili AI coding agenta? Opišite jednu odluku koju je predložio agent, kako ste provjerili ima li smisla i biste li danas tu odluku zadržali. Navedite i jedan dio projekta za koji smatrate da zahtijeva dodatnu provjeru, iako se kod izvršava bez pogreške.

Glazba mi je inače hobi, pa mi je projekt većinom legao. Neke stvari sam naučila, a dosta sam već prije znala osnove.ebala mi je pomoć samo oko toga kako se to konkretno piše u librosi (koje funkcije postoje, koji parametri, sintaksa).

Ono što sam ja napravila: većinu koda sam pisala sama, uz pomoć prijašnjih bilježnica i StackOverflowa, i dosta sam isprobavala. Ako nešto stvarno nije radilo, pitala bih AI da pregleda kod i vidi zašto ne radi. Uz to sam često tražila kratki prijedlog komentara i markdown teksta koji ide uz to, jer mi pisanje tih objašnjenja zna isprva biti teško. Tu prvu verziju sam koristila kao skicu, ne kao gotov tekst. Pitala sam za kratke, objektivne opise od jedne rečenice, koje sam popunila i zamijenila konkretnim opisom onoga što taj kod i/ili graf predstavlja. Strukturu prve verzije bih ponekad zadržala, ali sadržaj sam prepisala gotovo u cijelosti svojim riječima, uz konkretne brojke i poveznice koje agent nije mogao znati.

Jedna sitnica koju je agent predložio a nisam odmah očekivala da je bitna: umjesto da putanju do preuzetih podataka gradim ručno spajanjem stringova (path + "/"), predložio je path + os.sep. Prvo mi je djelovalo nebitno, ali objašnjenje je bilo da / radi na Macu i Linuxu, dok Windows koristi \, pa bi hardkodirani slash mogao zeznuti nekoga tko bilježnicu pokrene na drugom sustavu. Provjerila sam da os.sep stvarno vraća ispravan znak za trenutni sustav i zadržala to.

Dio kojem bih trebala posvetiti dodatnu pažnju, iako trenutno radi bez ijedne greške, je pretpostavka da redoslijed u genre_map (0 do 10) točno odgovara redoslijedu žanrova iz izvornog Kaggle opisa dataseta. To sam preuzela izravno iz naziva stupaca i nikad nisam formalno provjerila - ako je taj redoslijed makar malo drukčiji, sve oznake žanra u cijeloj bilježnici bi bile sustavno pogrešne, a kod bi svejedno radio glatko, bez ijednog upozorenja.

### 11. Kada biste morali nastaviti projekt, koje biste jedno poboljšanje napravili prvo? Odgovor treba obrazložiti prema očekivanom utjecaju na vjerodostojnost rezultata, a ne prema tome koliko je poboljšanje tehnički zanimljivo.

Prvo bih promijenila kako uopće definiram "točno" - umjesto da model mora pogoditi baš tu jednu oznaku žanra, računala bih predviđanje kao uspješno ako je stvarni žanr među top 2-3 najvjerojatnija predviđanja (top-k evaluacija).

Razlog: u pitanju 4 sama priznajem da žanr nije čvrsta istina i da ista pjesma opravdano može pripadati više kategorija. Međutim mjerim uspjeh i dalje strogo binarno, kao da tog problema nema. To znači da se možda ne mjeri koliko model razumije glazbu, nego koliko se slaže s proizvoljnom, ljudsko odlučenom odlukom da neka pjesma dobije baš tu jednu oznaku, a ne neku drugu jednako opravdanu. Top-k pristup bi mogao pokazati da model zapravo dobro "zna" da je pjesma negdje između Alt_Music i Indie Alt, samo ga trenutna metrika kažnjava jer nije pogodio točno onu jednu koju je netko drugi proizvoljno odabrao.

Ovo stavljam ispred kalibracije ili provjere genre_map reda jer te dvije stvari popravljaju kako mjerim postojeći broj, dok ovo mijenja što uopće smatram točnim odgovorom, na čemu se temelji svaki drugi broj u projektu.

## Zaključak

Slika projekta je otprilike sljedeća: audio značajke same po sebi nose stvaran signal za prepoznavanje žanra što se najbolje vidi kod žanrova s prepoznatljivim zvukom poput Instrumental ili Bollywood, gdje je model gotovo uvijek u pravu. Tamo gdje model ne uspijeva, kao kod Alt_Music, razlog nije nužno loš algoritam, nego kombinacija stvarnog zvučnog preklapanja među žanrovima i nesigurnosti u samim oznakama - žanr je kulturna kategorija, ne mjerljivo fizičko svojstvo, pa gornja granica točnosti nikad neće biti 100%, bez obzira koliko dobar model napravim.

Metodološki, najveći propust prve verzije bilježnice - korištenje istog validacijskog skupa za odabir modela i za izvještavanje rezultata, te računanje medijana prije podjele podataka - sam u ovoj verziji ispravila, i to se vidi u tome što je razlika između validacijske i test točnosti mala. To mi daje razumnu sigurnost da broj koji izvještavam nije slučajna sreća.

Ono što projekt trenutno ne pokriva, a trebao bi prije nego se ozbiljno koristi, jest provjera uploada prave audio datoteke na stvarnim primjerima s poznatim žanrom. Trenutno je to jedina veća funkcionalnost u bilježnici koja se oslanja isključivo na pretpostavku da je dovoljno dobra, bez ikakvog mjerenja. Da sam imala više vremena i pristup većem broju označenih audio datoteka, to bi bio prvi sljedeći korak, prije bilo kakvog eksperimentiranja s boljim modelima ili dodatnim značajkama.
