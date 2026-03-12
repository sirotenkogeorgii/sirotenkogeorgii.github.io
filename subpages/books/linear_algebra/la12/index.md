---
layout: default
title: Linear Algebra I & II
date: 2025-03-10
excerpt: Introduction, notation, complex numbers, polynomials, analytic geometry, and optimization.
tags:
  - linear-algebra
  - mathematics
---

# Linear Algebra I & II

## Kapitola 1 — Úvod

### O knize a lineární algebře

Lineární algebra je jedním ze základních matematických oborů. Její hlavní nástroje jsou vektory a matice. Vektory žijí v nějakém prostoru, a právě lineární algebra se zabývá lineárními objekty v prostoru — body, přímky, roviny apod. Studuje také lineární zobrazení (rotace, projekce atd.).

Matice představují data v maticové formě a odpovídají lineárním transformacím v eukleidovském prostoru. Na matice se lze dívat dvěma způsoby — algebraicky a geometricky.

Hlavní témata textu:

- **Soustavy lineárních rovnic** — nejzákladnější problém lineární algebry. Gaussova eliminace umí vyřešit principiálně každou soustavu.
- **Afinní podprostory** — každý útvar lze popsat pomocí soustavy rovnic.
- **Matice** — fundamentální nástroj. Typy matic a jejich vztah k soustavám lineárních rovnic.
- **Grupy a tělesa** — rozšíření výsledků na jiné číselné obory (komplexní čísla, $\mathbb{F}_2$, atd.).
- **Vektory a vektorové prostory** — axiomaticky zaváděné prostory, pojmy dimenze a isomorfismu.
- **Lineární zobrazení** — transformace zobrazující přímky na přímky a počátek na počátek.
- **Skalární součin** — geometrie (kolmost, vzdálenosti, projekce).
- **Determinanty a vlastní čísla** — charakteristiky matic (objem, explicitní vzorce, jemnější informace o chování matice).
- **Positivně definitní matice a kvadratické formy** — elipsoidy, optimalizace, statistika.
- **Maticové rozklady** — QR, SVD a další.

### Pojmy a značení

#### Číselné obory

- $\mathbb{N} = \lbrace 1, 2, \dots \rbrace$ — množina přirozených čísel
- $\mathbb{Z}$ — množina celých čísel
- $\mathbb{Q}$ — množina racionálních čísel
- $\mathbb{R}$ — množina reálných čísel
- $\mathbb{C}$ — množina komplexních čísel

#### Suma

Symbol $\sum$ reprezentuje součet všech instancí výrazu za sumou:

$$\sum_{i=1}^{n} a_i = a_1 + \ldots + a_n.$$

Součet přes prázdnou množinu indexů je definován jako $0$.

#### Produkt

Symbol $\prod$ se používá pro součin:

$$\prod_{i=1}^{n} a_i = a_1 \cdot a_2 \cdot \ldots \cdot a_n.$$

#### Kvantifikátory

- $\forall$ — "pro všechna" (univerzální kvantifikátor)
- $\exists$ — "existuje" (existenční kvantifikátor)

Například $\forall x \in \mathbb{R}: x + 1 \le e^x$ se čte "pro všechna reálná čísla $x$ platí nerovnost $x + 1 \le e^x$."

#### Modulo

Modulo $a \bmod n$ udává zbytek při dělení celého čísla $a$ přirozeným číslem $n$. Zbytek je definován jako číslo $b \in \lbrace 0, 1, \ldots, n-1 \rbrace$ takové, že $a = zn + b$ pro nějaké $z \in \mathbb{Z}$. Například:

$$17 \bmod 7 = 3, \qquad -17 \bmod 7 = 4.$$

#### Faktoriál

Faktoriál přirozeného čísla $n$ je součin $1 \cdot 2 \cdot 3 \cdot \ldots \cdot n$ a značí se $n!$.

#### Body a vektory

Bod, jakožto $n$-tice reálných čísel $(v_1, \ldots, v_n)$, se algebraicky chová stejně jako aritmetický vektor. Oba pojmy se interpretují různě až z geometrického hlediska, kdy nenulovému vektoru odpovídá určitý směr.

#### Zobrazení

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Zobrazení)</span></p>

Zobrazení $f$ z množiny $\mathcal{A}$ do množiny $\mathcal{B}$ se značí $f \colon \mathcal{A} \to \mathcal{B}$. Pro každé $a \in \mathcal{A}$ je $f(a)$ definováno a náleží do $\mathcal{B}$.

- **Prosté** (injektivní): $a_1 \neq a_2 \Rightarrow f(a_1) \neq f(a_2)$.
- **"Na"** (surjektivní): pro každé $b \in \mathcal{B}$ existuje $a \in \mathcal{A}$ takové, že $f(a) = b$.
- **Vzájemně jednoznačné** (bijekce): je prosté a "na".
- Je-li $f$ bijekce, pak existuje **inverzní zobrazení** $f^{-1} \colon \mathcal{B} \to \mathcal{A}$ definované $f^{-1}(b) = a$ pokud $f(a) = b$. Platí $(f^{-1})^{-1} = f$.
- **Složené zobrazení**: $(g \circ f)(a) = g(f(a))$. Složení bijekcí je bijekce. Skládání je asociativní: $h \circ (g \circ f) = (h \circ g) \circ f$.
- **Isomorfismus**: vzájemně jednoznačné zobrazení zachovávající strukturu. Existuje-li isomorfismus mezi $\mathcal{A}$ a $\mathcal{B}$, pak se množiny nazývají *isomorfní*.

</div>

#### Spočetná a nespočetná množina

Množina $M$ je **spočetná**, pokud existuje vzájemně jednoznačné zobrazení mezi $M$ a $\mathbb{N}$ nebo její podmnožinou. Spočetná je tak každá konečná množina nebo množiny $\mathbb{N}$, $\mathbb{Z}$, $\mathbb{Q}$. **Nespočetná** je množina, jež není spočetná — příkladem je $\mathbb{R}$.

#### Relace

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Relace)</span></p>

*Binární relace* $R$ na množině $M$ je libovolná podmnožina kartézského součinu $M \times M = \lbrace (x, y) ;\; x, y \in M \rbrace$. Relace $R$ je

- **reflexivní**, pokud $(x, x) \in R$ pro každé $x \in M$,
- **symetrická**, pokud $(x, y) \in R \Rightarrow (y, x) \in R$,
- **anti-symetrická**, pokud $(x, y) \in R$ a $(y, x) \in R \Rightarrow x = y$,
- **transitivní**, pokud $(x, y) \in R$ a $(y, z) \in R \Rightarrow (x, z) \in R$.

Relace je **ekvivalence**, pokud je reflexivní, symetrická a transitivní. Relace je **(částečné) uspořádání**, pokud je reflexivní, anti-symetrická a transitivní.

</div>

Příklady ekvivalence: rovnost čísel, rovnost zbytků při dělení, shodnost geometrických objektů, shoda barev.

Příklady částečného uspořádání: nerovnost $\le$, inkluze $\subseteq$, dělitelnost na přirozených číslech.

### Stavba matematické teorie

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Základní stavební jednotky)</span></p>

- **Definice** je přesné vymezení pojmu pomocí základních nebo dříve definovaných pojmů.
- **Tvrzení** je výrok, jehož pravdivost musí být stvrzena důkazem.
- **Věta** je význačnější tvrzení.
- **Lemma** je pomocné tvrzení sloužící k důkazu složitější věty.
- **Důsledek** je tvrzení, které víceméně jednoduše vyplývá z předchozího.
- **Důkaz** je posloupnost logických kroků formálně prokazující platnost tvrzení.

</div>

Tvrzení má tvar implikace "Pokud platí $\mathcal{P}$, potom platí $\mathcal{T}$". Dva základní důkazové postupy:

- **Důkaz přímý**: Z předpokladu $\mathcal{P}$ a posloupností platných odvození dojde k platnosti výroku $\mathcal{T}$.
- **Důkaz sporem**: Vyjde z předpokladu $\mathcal{P}$ a z negace $\mathcal{T}$ a dojde k logickému sporu.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Důkaz matematickou indukcí)</span></p>

Důkaz matematickou indukcí se používá pro tvrzení typu "Pro všechna přirozená $n$ platí $\mathcal{T}(n)$." Má dva kroky:

1. **Báze**: Nahlédneme platnost $\mathcal{T}(n)$ pro $n = 1$.
2. **Indukční krok** ($n \leftarrow n - 1$): Ukážeme platnost $\mathcal{T}(n)$ s využitím platnosti $\mathcal{T}(n-1)$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Důkaz indukcí)</span></p>

*Tvrzení:* Pro každé přirozené číslo $n$ je číslo $n^3 + 2n$ dělitelné třemi.

*Báze ($n = 1$):* $1^3 + 2 \cdot 1 = 3$, což je dělitelné třemi.

*Indukční krok:* Předpokládáme platnost pro $n-1$ a chceme ukázat platnost pro $n$:

$$n^3 + 2n = ((n-1)+1)^3 + 2((n-1)+1) = (n-1)^3 + 2(n-1) + 3(n-1)^2 + 3(n-1) + 3.$$

Z indukčního předpokladu je $(n-1)^3 + 2(n-1)$ dělitelné třemi, proto i $n^3 + 2n$ je dělitelné třemi.

</div>

### Reprezentace čísel

Vzhledem k omezené operační paměti nelze v počítači reprezentovat všechna reálná čísla. Čísla se standardně reprezentují v tzv. tvaru s pohyblivou řádovou čárkou

$$m \times b^t,$$

kde $b$ je základ číselné soustavy (většinou $b = 2$), $m$ je mantisa a $t$ exponent. Mantisa se normuje tak, aby $1 \le m < b$. Protože pro mantisu i exponent je na počítači omezená velikost, můžeme reprezentovat pouze konečně mnoho reálných čísel.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Zaokrouhlovací chyby)</span></p>

Zaokrouhlovací chyby se mohou akumulovat při vyhodnocování aritmetických operací. Například součet čísel $1/3$ a $1/3$ ve čtyřmístné dekadické aritmetice vede na součet reprezentací $0.3333$ a $0.3333$ s výsledkem $0.6666$. Nicméně nejbližší reprezentovatelné číslo ke skutečnému součtu $2/3$ je $0.6667$.

Vliv zaokrouhlovacích chyb musíme zohlednit při návrhu algoritmů. Touto problematikou se zabývá *numerická analýza*.

</div>

### Komplexní čísla

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Komplexní číslo)</span></p>

*Komplexní číslo* $z$ zavádíme jako výraz $a + bi$, kde $a, b \in \mathbb{R}$ a imaginární jednotka $i$ splňuje $i^2 = -1$. Zde $a$ je **reálná část** ($\operatorname{Re}(z)$) a $b$ je **imaginární část** ($\operatorname{Im}(z)$). Množinu komplexních čísel značíme $\mathbb{C}$.

</div>

Základní operace pro $z_1 = a + bi$, $z_2 = c + di$:

$$z_1 + z_2 = (a + c) + (b + d)i, \qquad z_1 z_2 = (ac - bd) + (cb + ad)i.$$

Pro $z_2 \neq 0$:

$$\frac{z_1}{z_2} = \frac{a + bi}{c + di} = \frac{a + bi}{c + di} \cdot \frac{c - di}{c - di} = \frac{ac + bd}{c^2 + d^2} + \frac{cb - ad}{c^2 + d^2}i.$$

Komplexní čísla mají geometrickou interpretaci: číslo $a + bi$ odpovídá bodu $(a, b)$ v komplexní (Gaussově) rovině.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Komplexně sdružené číslo a absolutní hodnota)</span></p>

Pro $z = a + bi$ definujeme:

- **Komplexně sdružené číslo**: $\overline{z} \coloneqq a - bi$.
- **Absolutní hodnota**: $\lvert z \rvert \coloneqq \sqrt{z \overline{z}} = \sqrt{a^2 + b^2}$.

Absolutní hodnota určuje eukleidovskou vzdálenost bodu $(a, b)$ od počátku. Komplexně sdružené číslo představuje překlopení podle reálné osy.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(Komplexně sdružená čísla a absolutní hodnoty)</span></p>

- $z = \overline{z}$ právě tehdy, když $z$ je reálné číslo,
- $z + \overline{z} = 2\operatorname{Re}(z)$,
- $\overline{z_1 + z_2} = \overline{z_1} + \overline{z_2}$,
- $\overline{z_1 \cdot z_2} = \overline{z_1} \cdot \overline{z_2}$,
- $\lvert z_1 + z_2 \rvert \le \lvert z_1 \rvert + \lvert z_2 \rvert$ (trojúhelníková nerovnost),
- $\lvert z_1 \cdot z_2 \rvert = \lvert z_1 \rvert \cdot \lvert z_2 \rvert$,
- $\operatorname{Re}(z) \le \lvert z \rvert$.

Pozor: obecně $\lvert z \rvert^2 \neq z^2$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Geometrický význam operací)</span></p>

- **Sčítání** $z \mapsto z + v$ představuje posun ve směru vektoru $(\operatorname{Re}(v), \operatorname{Im}(v))$.
- **Násobení** $z \mapsto vz$: je-li $v$ reálné, jedná se o škálování s násobkem $\lvert v \rvert$. Je-li $v$ komplexní a $\lvert v \rvert = 1$, jedná se o otočení o úhel $\alpha$, který $v$ svírá s reálnou osou. V obecném případě se kombinují obě vlastnosti.

Příklad: $v = i$ představuje otočení o $90°$.

</div>

### Polynomy

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Polynom)</span></p>

*Reálným polynomem* stupně $n$ je funkce $p(x) = a_n x^n + a_{n-1} x^{n-1} + \ldots + a_1 x + a_0$, kde $a_0, \ldots, a_n \in \mathbb{R}$ a $a_n \neq 0$. Kromě reálných polynomů lze uvažovat polynomy s komplexními koeficienty.

</div>

Operace s polynomy $p(x) = a_n x^n + \ldots + a_1 x + a_0$ a $q(x) = b_m x^m + \ldots + b_1 x + b_0$ (nechť $n \ge m$):

- **Sčítání**: $p(x) + q(x) = a_n x^n + \ldots + (a_m + b_m) x^m + \ldots + (a_0 + b_0)$.
- **Násobení**: $p(x)q(x) = a_n b_m x^{n+m} + \ldots + a_0 b_0$.
- **Dělení se zbytkem**: Existuje jednoznačně určený polynom $r(x)$ stupně $n - m$ a polynom $s(x)$ stupně menšího než $m$ tak, že $p(x) = r(x)q(x) + s(x)$.

#### Kořeny

*Kořen* polynomu $p(x)$ je taková hodnota $x^* \in \mathbb{R}$ (resp. $\mathbb{C}$), že $p(x^*) = 0$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Základní věta algebry)</span></p>

Každý polynom s komplexními koeficienty má alespoň jeden komplexní kořen.

</div>

Je-li $x_1$ kořen polynomu $p(x)$, pak $p(x)$ je dělitelný členem $(x - x_1)$ beze zbytku a podíl je polynom stupně $n - 1$. Opakovanou aplikací základní věty algebry dostaneme rozklad

$$p(x) = a_n (x - x_1)(x - x_2) \cdots (x - x_n),$$

kde $x_1, \ldots, x_n$ jsou kořeny (započítáváme i násobnosti). Polynom stupně $n$ má tedy právě $n$ kořenů.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Hledání kořenů)</span></p>

Kořeny polynomu druhého stupně $a_2 x^2 + a_1 x + a_0$ najdeme vzorečkem $x_{1,2} = \frac{1}{2a_2}(-a_1 \pm \sqrt{a_1^2 - 4a_2 a_0})$. Pro třetí stupeň existují Cardanovy vzorce (ale jsou mnohem komplikovanější). Abel (1824) ukázal, že pro polynomy stupňů vyšších než 4 žádný vzoreček na výpočet kořenů neexistuje. Kořeny tedy hledáme iteračními metodami.

</div>

### Analytická geometrie

#### Přímka v rovině

*Rovnicový popis* přímky v rovině:

$$a_1 x_1 + a_2 x_2 = b,$$

kde $a_1, a_2, b \in \mathbb{R}$ a alespoň jedno z $a_1, a_2$ je nenulové. Všechny body $(x_1, x_2)$ splňující rovnici představují přímku. Vektoru $(a_1, a_2)$ se říká **normálový vektor** a je kolmý na přímku.

*Parametrický popis* přímky v rovině:

$$(x_1, x_2) = (b_1, b_2) + t \cdot (v_1, v_2), \quad t \in \mathbb{R},$$

kde $(b_1, b_2)$ je daný bod přímky a $(v_1, v_2) \neq (0, 0)$ je **směrový vektor** přímky.

#### Přímka v prostoru

Parametrický popis přímky v prostoru:

$$(x_1, x_2, x_3) = (b_1, b_2, b_3) + t \cdot (v_1, v_2, v_3), \quad t \in \mathbb{R},$$

kde $(v_1, v_2, v_3) \neq (0, 0, 0)$ je směrový vektor. V $n$-dimenzionálním prostoru analogicky:

$$(x_1, \ldots, x_n) = (b_1, \ldots, b_n) + t \cdot (v_1, \ldots, v_n), \quad t \in \mathbb{R}.$$

Směrový vektor je určen až na násobek jednoznačně, zatímco bod $(b_1, \ldots, b_n)$ lze zvolit na přímce libovolně.

#### Rovina v prostoru

Jedna rovnice $a_1 x_1 + a_2 x_2 + a_3 x_3 = b$, kde $(a_1, a_2, a_3) \neq (0, 0, 0)$, popisuje v prostoru rovinu. Vektor $(a_1, a_2, a_3)$ je její **normálový vektor** — je kolmý na rovinu a je určen až na násobek jednoznačně.

Rovnicový popis přímky v prostoru vyžaduje dvě rovnice:

$$a_1 x_1 + a_2 x_2 + a_3 x_3 = b_1, \qquad a_1' x_1 + a_2' x_2 + a_3' x_3 = b_2,$$

přičemž normály musí být nenulové a nesmí udávat stejný směr (roviny nesmí být rovnoběžné).

### Optimalizace

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Optimalizační úloha)</span></p>

Buď $f \colon \mathbb{R}^n \to \mathbb{R}$ reálná funkce a $M \subseteq \mathbb{R}^n$ množina bodů. Úloha *optimalizace* je

$$\min f(x) \quad \text{za podmínky } x \in M.$$

Hledáme takový bod $x \in M$, že $f(x) \le f(y)$ pro všechna $y \in M$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Lineární optimalizace)</span></p>

Uvažujme funkci $f(x_1, x_2) = x_1 - x_2$ a množinu $M$ v rovině zadanou omezeními $x_1 + 2x_2 \le 4$, $x_1 \ge 0$, $x_2 \ge 0$. Množina $M$ představuje trojúhelník s vrcholy $(0, 0)$, $(4, 0)$ a $(0, 2)$. Minimální hodnota funkce se nabyde v bodě $(0, 2)$.

</div>

### Matematický software

Funkce na řešení základních úloh lineární algebry jsou standardní součástí matematických softwarových systémů:

- **Matlab** — bohaté prostředí pro numerické výpočty. **Octave** je open-source alternativa s téměř identickou syntaxí.
- **Mathematica** a **Maple** — přední systémy pro symbolické výpočty. **SageMath** je volně dostupná alternativa.
- **Julia** — moderní jazyk pro výpočetně náročné úlohy s podporou paralelních a distribuovaných výpočtů.
- **Wolfram Alpha** — on-line výpočetní systém založený na Mathematice.

## Kapitola 2 — Soustavy lineárních rovnic

### Základní pojmy

Soustavy lineárních rovnic patří mezi základní algebraické úlohy a setkáme se s nimi skoro všude — pokud nějaký problém nevede na soustavu rovnic přímo, tak se soustavy rovnic často objeví jako jeho podproblém.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Historická úloha — Chiu-chang Suan-shu, ca 200 př. n. l.)</span></p>

*Tři snopy dobrého obilí, dva snopy průměrného a jeden podřadného se prodávají celkem za 39 dou. Dva snopy dobrého obilí, tři průměrného a jeden podřadného se prodávají za 34 dou. Jeden snop dobrého obilí, dva průměrného a tři podřadného se prodávají za 26 dou. Jaká je cena za jeden snop dobrého / průměrného / podřadného obilí?*

Zapsáno dnešní matematikou dostáváme soustavu rovnic:

$$3x + 2y + z = 39, \qquad 2x + 3y + z = 34, \qquad x + 2y + 3z = 26,$$

kde $x, y, z$ jsou neznámé pro ceny za jeden snop dobrého / průměrného / podřadného obilí.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Matice)</span></p>

Reálná *matice* typu $m \times n$ je obdélníkové schema (tabulka) reálných čísel

$$A = \begin{pmatrix} a_{11} & a_{12} & \ldots & a_{1n} \\ \vdots & \vdots & & \vdots \\ a_{m1} & a_{m2} & \ldots & a_{mn} \end{pmatrix}.$$

Prvek na pozici $(i, j)$ matice $A$ (tj. v $i$-tém řádku a $j$-tém sloupci) značíme $a_{ij}$ nebo $A_{ij}$. Množinu všech reálných matic typu $m \times n$ značíme $\mathbb{R}^{m \times n}$; podobně pro komplexní, racionální atd. Je-li $m = n$, matici nazýváme *čtvercovou*.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Vektor)</span></p>

Reálný $n$-rozměrný aritmetický sloupcový *vektor* je matice typu $n \times 1$:

$$x = \begin{pmatrix} x_1 \\ \vdots \\ x_n \end{pmatrix}$$

a řádkový vektor je matice typu $1 \times n$: $x = (x_1, \ldots, x_n)$.

Standardně, pokud není řečeno jinak, uvažujeme vektory sloupcové. Množina všech $n$-rozměrných vektorů se značí $\mathbb{R}^n$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Notace pro řádky a sloupce)</span></p>

- $i$-tý řádek matice $A$ se značí $A_{i*} = (a_{i1}, a_{i2}, \ldots, a_{in})$.
- $j$-tý sloupec matice $A$ se značí $A_{*j} = (a_{1j}, a_{2j}, \ldots, a_{mj})^T$.

Matici $A \in \mathbb{R}^{m \times n}$ lze tudíž rozepsat po sloupcích $A = (A_{*1} \; A_{*2} \; \ldots \; A_{*n})$ nebo po řádcích. Obecné matice značíme velkými písmeny a vektory malými písmeny.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Soustava lineárních rovnic)</span></p>

Mějme soustavu $m$ lineárních rovnic o $n$ neznámých:

$$a_{11}x_1 + a_{12}x_2 + \ldots + a_{1n}x_n = b_1, \quad \ldots, \quad a_{m1}x_1 + a_{m2}x_2 + \ldots + a_{mn}x_n = b_m,$$

kde $a_{ij}, b_i$ jsou dané koeficienty a $x_1, \ldots, x_n$ jsou neznámé. *Řešením* rozumíme každý vektor $x \in \mathbb{R}^n$ vyhovující všem rovnicím.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Matice soustavy a rozšířená matice soustavy)</span></p>

*Matice soustavy* je matice

$$A = \begin{pmatrix} a_{11} & a_{12} & \ldots & a_{1n} \\ \vdots & \vdots & & \vdots \\ a_{m1} & a_{m2} & \ldots & a_{mn} \end{pmatrix}$$

a *rozšířená matice soustavy* je

$$(A \mid b) = \begin{pmatrix} a_{11} & a_{12} & \ldots & a_{1n} & b_1 \\ a_{21} & a_{22} & \ldots & a_{2n} & b_2 \\ \vdots & \vdots & & \vdots & \vdots \\ a_{m1} & a_{m2} & \ldots & a_{mn} & b_m \end{pmatrix}.$$

Svislá čára v rozšířené matici symbolizuje rovnost mezi levou a pravou stranou soustavy. Řádky odpovídají rovnicím, sloupce nalevo proměnným $x_1, \ldots, x_n$ a poslední sloupec hodnotám na pravé straně.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Geometrický význam soustavy rovnic)</span></p>

Pro $n = 2$ (dvě rovnice o dvou neznámých) popisuje každá rovnice přímku v rovině $\mathbb{R}^2$. Řešení soustavy leží v průniku obou přímek.

Pro $n = 3$ popisuje každá rovnice rovinu v prostoru $\mathbb{R}^3$. Řešení soustavy je průnik těchto rovin — může to být jediný bod (obecná poloha), přímka (roviny obsahují společnou přímku), nebo prázdná množina (rovnoběžné roviny).

Obecně pro libovolné $n$ rovnice určují tzv. nadroviny a řešení soustavy hledáme v jejich průniku.

</div>

### Elementární řádkové úpravy

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Elementární řádkové úpravy)</span></p>

Elementární řádkové úpravy matice jsou:

1. Vynásobení $i$-tého řádku reálným číslem $\alpha \neq 0$ (tj. vynásobí se všechny prvky řádku).
2. Přičtení $\alpha$-násobku $j$-tého řádku k $i$-tému, přičemž $i \neq j$ a $\alpha \in \mathbb{R}$.
3. Výměna $i$-tého a $j$-tého řádku.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Nezávislost úprav)</span></p>

Ve skutečnosti výše zmíněné úpravy nejsou zas tak elementární. U druhé řádkové úpravy vystačíme jen s $\alpha = 1$ a třetí úpravu lze simulovat pomocí předchozích dvou.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Tvrzení 2.10)</span></p>

Elementární řádkové operace zachovávají množinu řešení soustavy.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Idea důkazu</summary>

Základní myšlenkou je ukázat, že elementární úpravou se množina řešení nemění. Elementární úpravou neztratíme žádné řešení, protože pokud je $x$ řešením před úpravou, je i po úpravě. A naopak, úpravou žádné řešení nepřibyde, protože každá úprava má svoji inverzní úpravu — vhodnou elementární úpravou můžeme dojít zpět k původnímu tvaru soustavy.

</details>
</div>

### Gaussova eliminace

Základní myšlenka metody je transformace rozšířené matice soustavy pomocí elementárních řádkových úprav na jednodušší matici, ze které řešení snadno vyčteme.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Gaussova eliminace — ukázka)</span></p>

Uvažujme soustavu lineárních rovnic:

$$x_1 + 2x_2 + 3x_3 = 32, \quad x_1 + x_2 + 2x_3 = 21, \quad 3x_1 + x_2 + 3x_3 = 35.$$

**Dopředná eliminace:** Postupně eliminujeme proměnné. Odečtením první rovnice od druhé a trojnásobku od třetí, pak dalšími úpravami dostaneme:

$$x_1 + 2x_2 + 3x_3 = 32, \quad -x_2 - x_3 = -11, \quad -x_3 = -6.$$

**Zpětná substituce:** Z třetí rovnice $x_3 = 6$. Dosadíme do druhé: $x_2 = 5$. Dosadíme do první: $x_1 = 4$. Řešení soustavy je $(x_1, x_2, x_3) = (4, 5, 6)$.

</div>

#### Odstupňovaný tvar matice (REF)

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Odstupňovaný tvar matice — REF)</span></p>

Matice $A \in \mathbb{R}^{m \times n}$ je v *řádkově odstupňovaném tvaru* (row echelon form, REF), pokud existuje $r$ takové, že:

- řádky $1, \ldots, r$ jsou nenulové (tj. každý obsahuje aspoň jednu nenulovou hodnotu),
- řádky $r+1, \ldots, m$ jsou nulové,

a navíc, označíme-li $p_i = \min\lbrace j;\; a_{ij} \neq 0 \rbrace$ pozici prvního nenulového prvku v $i$-tém řádku, tak platí $p_1 < p_2 < \cdots < p_r$.

Pozice $(1, p_1), (2, p_2), \ldots, (r, p_r)$ se nazývají **pivoty**. Sloupce $p_1, p_2, \ldots, p_r$ se nazývají **bázické** a ostatní sloupce **nebázické**.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Matice v REF a ne-REF)</span></p>

V odstupňovaném tvaru jsou matice:

$$\begin{pmatrix} 1 & 2 & 3 \\ 0 & 4 & 5 \\ 0 & 0 & 6 \end{pmatrix}, \quad \begin{pmatrix} 1 & 2 & 3 \\ 0 & 0 & 5 \\ 0 & 0 & 0 \end{pmatrix}, \quad \begin{pmatrix} 1 & 0 \\ 0 & 2 \\ 0 & 0 \end{pmatrix}, \quad \begin{pmatrix} 0 & 1 & 2 & 3 \\ 0 & 0 & 4 & 5 \\ 0 & 0 & 0 & 0 \end{pmatrix}.$$

V odstupňovaném tvaru *nejsou*:

$$\begin{pmatrix} 1 & 1 & 1 \\ 0 & 0 & 2 \\ 0 & 0 & 3 \end{pmatrix}, \quad \begin{pmatrix} 1 & 2 & 3 \\ 2 & 3 & 0 \\ 3 & 0 & 0 \end{pmatrix}, \quad \begin{pmatrix} 0 \\ 1 \\ 0 \end{pmatrix}, \quad \begin{pmatrix} 0 & 0 & 0 & 1 \\ 0 & 0 & 2 & 2 \\ 0 & 0 & 0 & 0 \end{pmatrix}.$$

</div>

#### Hodnost matice

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Hodnost matice)</span></p>

*Hodností* matice $A$ rozumíme počet nenulových řádků po převodu do odstupňovaného tvaru a značíme $\operatorname{rank}(A)$.

</div>

Hodnost matice je tedy rovna počtu pivotů (tj. číslu $r$) po převedení do odstupňovaného tvaru. I když odstupňovaný tvar není jednoznačný, pozice pivotů jednoznačné jsou (viz Věta 2.28 níže). Proto je pojem hodnosti dobře definován.

#### Algoritmus REF

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(REF(A) — Algoritmus 2.15)</span></p>

**Vstup:** matice $A \in \mathbb{R}^{m \times n}$.

1. $i := 1$, $j := 1$.
2. **if** $a_{k\ell} = 0$ pro všechna $k \ge i$ a $\ell \ge j$ **then** konec.
3. $j := \min\lbrace \ell;\; \ell \ge j, \; a_{k\ell} \neq 0 \text{ pro nějaké } k \ge i \rbrace$ (přeskočíme nulové podsloupečky).
4. Urči $k$ takové, že $a_{kj} \neq 0$, $k \ge i$ a vyměň řádky $A_{i*}$ a $A_{k*}$.
5. Pro všechna $k > i$ polož $A_{k*} := A_{k*} - \frac{a_{kj}}{a_{ij}} A_{i*}$ (2. elementární úprava).
6. Polož $i := i + 1$, $j := j + 1$, a jdi na krok 2.

**Výstup:** matice $A$ v odstupňovaném tvaru.

V praxi se v kroku 4 doporučuje tzv. **parciální pivotizace** — zvolit kandidáta $a_{kj}$ s maximální absolutní hodnotou, což má lepší numerické vlastnosti.

</div>

Algoritmus převádí matici do odstupňovaného tvaru po nejvýše $\min(m, n)$ iteracích hlavního cyklu. V kroku 5 vynulujeme všechny prvky pod pivotem $(i, j)$.

#### Gaussova eliminace — řešení soustavy

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Gaussova eliminace — Algoritmus 2.17)</span></p>

**Vstup:** soustava rovnic $(A \mid b)$, kde $A \in \mathbb{R}^{m \times n}$, $b \in \mathbb{R}^m$.

Převedeme rozšířenou matici soustavy $(A \mid b)$ na odstupňovaný tvar $(A' \mid b')$ a označíme $r = \operatorname{rank}(A \mid b)$. Nastane právě jedna ze tří situací:

**(A) Soustava nemá řešení.** Nastane v případě, že poslední sloupec je bázický, čili $\operatorname{rank}(A) < \operatorname{rank}(A \mid b)$. Poslední nenulový řádek má tvar $0x_1 + \ldots + 0x_n = b_r' \neq 0$.

**(B1) Soustava má právě jedno řešení.** Nastane pokud $r = n$ (počet proměnných je roven počtu pivotů) a poslední sloupec je nebázický. Řešení najdeme **zpětnou substitucí**: Postupně pro $k = n, n-1, \ldots, 1$ dosadíme

$$x_k := \frac{b_k' - \sum_{j=k+1}^{n} a_{kj}' x_j}{a_{kk}'}.$$

**(B2) Soustava má nekonečně mnoho řešení.** Nastane pokud $r < n$. V matici je alespoň jeden nebázický sloupec. Nebázické proměnné jsou volné parametry, bázické proměnné dopočítáme zpětnou substitucí. Počet nebázických proměnných $n - r > 0$ vyjadřuje dimenzi množiny řešení.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Gaussova eliminace s nekonečně mnoha řešeními)</span></p>

Vyřešíme soustavu s rozšířenou maticí:

$$\begin{pmatrix} 2 & 2 & -1 & 5 & 1 \\ 4 & 5 & 0 & 9 & 3 \\ 0 & 1 & 2 & 2 & 4 \\ 2 & 4 & 3 & 7 & 7 \end{pmatrix} \xrightarrow{\text{REF}} \begin{pmatrix} 2 & 2 & -1 & 5 & 1 \\ 0 & 1 & 2 & -1 & 1 \\ 0 & 0 & 0 & 3 & 3 \\ 0 & 0 & 0 & 0 & 0 \end{pmatrix}.$$

Bázické sloupce: 1, 2, 4. Nebázický sloupec: 3 ($x_3$ je volná proměnná). Zpětná substituce:

1. $x_4 = 1$,
2. $x_3$ je volná (nebázická) proměnná,
3. $x_2 = 1 + x_4 - 2x_3 = 2 - 2x_3$,
4. $x_1 = \frac{1}{2}(1 - 5x_4 + x_3 - 2x_2) = -4 + \frac{5}{2}x_3$.

Všechna řešení: $(-4 + \tfrac{5}{2}x_3,\; 2 - 2x_3,\; x_3,\; 1)$, kde $x_3 \in \mathbb{R}$, neboli

$$(-4, 2, 0, 1) + x_3 \cdot (\tfrac{5}{2}, -2, 1, 0), \quad x_3 \in \mathbb{R}.$$

Množina řešení představuje přímku v $\mathbb{R}^4$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Výpočetní složitost)</span></p>

Počet operací algoritmu REF$(A)$ se dá vyjádřit jako polynom proměnné $n$. Hlavní člen je $n^2$ součinů a $n^2$ odčítání v prvním cyklu, $(n-1)^2$ v druhém atd. S využitím vzorečku $\sum_{k=1}^{n} k^2 = \frac{1}{6}n(n+1)(2n+1)$ vidíme, že celková asymptotická složitost algoritmu REF je řádově $\frac{2}{3}n^3$ operací.

Zpětná substituce počítá řádově $n^2$ operací, proto na celkovou složitost nemá zásadní vliv.

</div>

### Gaussova–Jordanova eliminace

#### Redukovaný odstupňovaný tvar (RREF)

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Redukovaný odstupňovaný tvar matice — RREF)</span></p>

Matice $A \in \mathbb{R}^{m \times n}$ je v *redukovaném řádkově odstupňovaném tvaru* (reduced row echelon form, RREF), pokud je v REF tvaru a navíc platí:

- $a_{1p_1} = a_{2p_2} = \ldots = a_{rp_r} = 1$, tedy na pozicích pivotů jsou jedničky, a
- pro každé $i = 1, \ldots, r$ je $a_{1p_i} = a_{2p_i} = \ldots = a_{i-1,p_i} = 0$, tedy nad každým pivotem jsou samé nuly.

</div>

#### Algoritmus RREF

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(RREF(A) — Algoritmus 2.21)</span></p>

**Vstup:** matice $A \in \mathbb{R}^{m \times n}$.

1. $i := 1$, $j := 1$.
2. **if** $a_{k\ell} = 0$ pro všechna $k \ge i$ a $\ell \ge j$ **then** konec.
3. $j := \min\lbrace \ell;\; \ell \ge j, \; a_{k\ell} \neq 0 \text{ pro nějaké } k \ge i \rbrace$.
4. Urči $a_{kj} \neq 0$, $k \ge i$ a vyměň řádky $A_{i*}$ a $A_{k*}$.
5. Polož $A_{i*} := \frac{1}{a_{ij}} A_{i*}$ (nyní je na pozici pivota hodnota $1$).
6. Pro všechna $k \neq i$ polož $A_{k*} := A_{k*} - a_{kj} A_{i*}$ (2. elementární úprava — eliminuje i **nad** pivotem).
7. Polož $i := i + 1$, $j := j + 1$, a jdi na krok 2.

**Výstup:** matice $A$ v redukovaném odstupňovaném tvaru.

</div>

Rozdíl oproti REF algoritmu je v krocích 5 a 6: pivoty se normují na jedničku a eliminuje se nejen pod, ale i nad pivotem.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Převod na RREF)</span></p>

$$\begin{pmatrix} 2 & 2 & -1 & 5 \\ 4 & 5 & 0 & 9 \\ 0 & 1 & 2 & 2 \\ 2 & 4 & 3 & 7 \end{pmatrix} \xrightarrow{\text{RREF}} \begin{pmatrix} 1 & 0 & -2.5 & 0 \\ 0 & 1 & 2 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 0 & 0 \end{pmatrix}.$$

</div>

#### Gaussova–Jordanova eliminace — řešení soustavy

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Gaussova–Jordanova eliminace — Algoritmus 2.23)</span></p>

**Vstup:** soustava rovnic $(A \mid b)$, kde $A \in \mathbb{R}^{m \times n}$, $b \in \mathbb{R}^m$.

Převedeme rozšířenou matici soustavy na redukovaný odstupňovaný tvar $(A' \mid b')$ a označíme $r = \operatorname{rank}(A \mid b)$. Rozlišíme tři situace:

**(A) Soustava nemá řešení.** Nastane pokud poslední sloupec je bázický, čili $\operatorname{rank}(A) < \operatorname{rank}(A \mid b)$.

**(B1) Soustava má právě jedno řešení.** Nastane pokud poslední sloupec je nebázický a zároveň $r = n$. Všechny sloupce $1, \ldots, n$ jsou bázické a řešení je přímo $(x_1, \ldots, x_n) = (b_1', \ldots, b_n')$.

**(B2) Soustava má nekonečně mnoho řešení.** Nastane pokud $r < n$. Nebázické proměnné $x_i$, $i \in N = \lbrace 1, \ldots, n \rbrace \setminus \lbrace p_1, \ldots, p_r \rbrace$, jsou volné parametry. Bázické proměnné dopočítáme zpětnou substitucí:

$$x_{p_k} := b_k' - \sum_{j \in N,\; j > p_k} a_{kj}' x_j.$$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Gaussova–Jordanova eliminace)</span></p>

Soustava z předchozího příkladu řešená Gaussovou–Jordanovou eliminací:

$$\begin{pmatrix} 2 & 2 & -1 & 5 & 1 \\ 4 & 5 & 0 & 9 & 3 \\ 0 & 1 & 2 & 2 & 4 \\ 2 & 4 & 3 & 7 & 7 \end{pmatrix} \xrightarrow{\text{RREF}} \begin{pmatrix} 1 & 0 & -2.5 & 0 & -4 \\ 0 & 1 & 2 & 0 & 2 \\ 0 & 0 & 0 & 1 & 1 \\ 0 & 0 & 0 & 0 & 0 \end{pmatrix}.$$

Kroky zpětné substituce: $x_4 = 1$, $x_3$ volná, $x_2 = 2 - 2x_3$, $x_1 = -4 + \frac{5}{2}x_3$. Stejný výsledek jako Gaussovou eliminací.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Gaussova vs. Gaussova–Jordanova eliminace)</span></p>

Obě metody mají stejnou asymptotickou složitost řádově $n^3$. Gaussova eliminace je přibližně o třetinu rychlejší, na druhou stranu Gaussova–Jordanova eliminace (resp. RREF tvar) je potřeba při invertování matic.

Výpočetní složitost RREF$(A)$ pro čtvercovou matici řádu $n$ je řádově $n^3$ aritmetických operací.

</div>

#### Jednoznačnost RREF

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Jednoznačnost RREF — Věta 2.28)</span></p>

RREF tvar matice je jednoznačný. Bez ohledu na to, jaké elementární řádkové úpravy a v jakém pořadí vykonáváme, výsledný RREF tvar matice je vždy tentýž.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Idea důkazu</summary>

Pro spor předpokládejme, že matice $A \in \mathbb{R}^{m \times n}$ má dva různé RREF tvary $A_1$ a $A_2$. Označme $i$ index prvního sloupce, ve kterém se $A_1, A_2$ liší. Odstraníme z matic $A, A_1, A_2$ všechny nebázické sloupce před $i$-tým. Výsledné matice $B, B_1, B_2$ mají RREF tvary $B_1$ a $B_2$. Pokud interpretujeme matici $B$ jako soustavu lineárních rovnic, z RREF tvaru $B_1$ vyčteme jiné řešení než z tvaru $B_2$, což je spor. Pokud obě soustavy jsou neřešitelné, pak jejich poslední sloupce $c, d$ jsou bázické, a proto stejné.

</details>
</div>

#### Frobeniova věta

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Frobeniova věta — Poznámka 2.25)</span></p>

Soustava $(A \mid b)$ má alespoň jedno řešení právě tehdy, když $\operatorname{rank}(A) = \operatorname{rank}(A \mid b)$.

</div>

### Aplikace

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Elektrický obvod)</span></p>

Uvažujme elektrický obvod s odpory $10\,\Omega$, $10\,\Omega$, $20\,\Omega$ a zdroji $10\,\text{V}$, $5\,\text{V}$. Chceme určit proudy $I_1, I_2, I_3$. Pomocí Kirchhoffových zákonů (zákon o proudu: $I_1 + I_2 - I_3 = 0$; zákon o napětí pro smyčky) dostaneme soustavu:

$$\begin{pmatrix} 1 & 1 & -1 & 0 \\ 10 & -10 & 0 & 10 \\ 0 & 10 & 20 & 5 \end{pmatrix}.$$

Vyřešením máme $I_1 = 0.7\,\text{A}$, $I_2 = -0.3\,\text{A}$, $I_3 = 0.4\,\text{A}$.

</div>

## Kapitola 3 — Matice

Matice jsme zavedli v minulé kapitole ke kompaktnímu zápisu soustav lineárních rovnic a popisu metod na jejich řešení. Matice však mají mnohem širší využití, a proto se na ně v této kapitole podíváme podrobněji. Zavedeme několik typů matic a základní operace, které umožní s maticemi lépe a jednodušeji zacházet.

### Základní operace s maticemi

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Rovnost, součet a násobek matic)</span></p>

- **Rovnost:** Dvě matice se rovnají, $A = B$, pokud mají stejné rozměry $m \times n$ a $A_{ij} = B_{ij}$ pro $i = 1, \ldots, m$, $j = 1, \ldots, n$.
- **Součet:** Buď $A, B \in \mathbb{R}^{m \times n}$. Pak $A + B$ je matice typu $m \times n$ s prvky $(A + B)_{ij} = A_{ij} + B_{ij}$.
- **Násobek skalárem:** Buď $\alpha \in \mathbb{R}$ a $A \in \mathbb{R}^{m \times n}$. Pak $\alpha A$ je matice typu $m \times n$ s prvky $(\alpha A)_{ij} = \alpha A_{ij}$.

Odčítání definujeme přirozeně jako $A - B := A + (-1)B$. Speciální maticí je **nulová matice**, jejíž všechny prvky jsou nuly; značíme ji $0$ či $0_{m \times n}$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(Součet a násobky matic)</span></p>

Pro reálná čísla $\alpha, \beta$ a matice $A, B, C \in \mathbb{R}^{m \times n}$ platí:

1. $A + B = B + A$ (komutativita),
2. $(A + B) + C = A + (B + C)$ (asociativita),
3. $A + 0 = A$,
4. $A + (-1)A = 0$,
5. $\alpha(\beta A) = (\alpha \beta) A$,
6. $1A = A$,
7. $\alpha(A + B) = \alpha A + \alpha B$ (distributivita),
8. $(\alpha + \beta)A = \alpha A + \beta A$ (distributivita).

</div>

#### Maticový součin

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Součin matic)</span></p>

Buď $A \in \mathbb{R}^{m \times p}$ a $B \in \mathbb{R}^{p \times n}$. Pak $AB$ je matice typu $m \times n$ s prvky

$$(AB)_{ij} = \sum_{k=1}^{p} A_{ik} B_{kj}.$$

Prvek na pozici $(i, j)$ součinu $AB$ spočítáme jako skalární součin $i$-tého řádku matice $A$ a $j$-tého sloupce matice $B$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Jednotková matice)</span></p>

*Jednotková matice* řádu $n$, značená $I$ nebo $I_n$, je čtvercová matice s prvky $I_{ij} = 1$ pro $i = j$ a $I_{ij} = 0$ jinak. Má jedničky na diagonále a nuly jinde. *Jednotkový vektor* $e_i$ je $i$-tý sloupec jednotkové matice, tj. $e_i = I_{*i}$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(Součin matic)</span></p>

Pro číslo $\alpha$ a matice $A, B, C$ vhodných rozměrů platí:

1. $(AB)C = A(BC)$ (asociativita),
2. $A(B + C) = AB + AC$ (distributivita zleva),
3. $(A + B)C = AC + BC$ (distributivita zprava),
4. $\alpha(AB) = (\alpha A)B = A(\alpha B)$,
5. $0A = A0 = 0$,
6. $I_m A = A I_n = A$ kde $A \in \mathbb{R}^{m \times n}$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Důkaz asociativity</summary>

Buď $A \in \mathbb{R}^{m \times p}$, $B \in \mathbb{R}^{p \times r}$ a $C \in \mathbb{R}^{r \times n}$. Pak $AB$ má typ $m \times r$, $BC$ má typ $p \times n$ a oba součiny $(AB)C$, $A(BC)$ mají typ $m \times n$. Na pozici $(i, j)$:

$$((AB)C)_{ij} = \sum_{k=1}^{r} (AB)_{ik} C_{kj} = \sum_{k=1}^{r} \left(\sum_{\ell=1}^{p} A_{i\ell} B_{\ell k}\right) C_{kj} = \sum_{k=1}^{r} \sum_{\ell=1}^{p} A_{i\ell} B_{\ell k} C_{kj},$$

$$(A(BC))_{ij} = \sum_{\ell=1}^{p} A_{i\ell} (BC)_{\ell j} = \sum_{\ell=1}^{p} A_{i\ell} \left(\sum_{k=1}^{r} B_{\ell k} C_{kj}\right) = \sum_{\ell=1}^{p} \sum_{k=1}^{r} A_{i\ell} B_{\ell k} C_{kj}.$$

Oba výrazy jsou shodné díky komutativitě sčítání reálných čísel.

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Nekomutativita součinu)</span></p>

Součin matic obecně není komutativní: pro mnoho matic je $AB \neq BA$. Například pro $A = \bigl(\begin{smallmatrix} 0 & 1 \\ 0 & 0 \end{smallmatrix}\bigr)$, $B = \bigl(\begin{smallmatrix} 1 & 0 \\ 0 & 0 \end{smallmatrix}\bigr)$ je $AB = \bigl(\begin{smallmatrix} 0 & 0 \\ 0 & 0 \end{smallmatrix}\bigr)$ ale $BA = \bigl(\begin{smallmatrix} 0 & 1 \\ 0 & 0 \end{smallmatrix}\bigr)$.

Navíc se může stát, že součin $AB$ má smysl, ale $BA$ nikoli (matice mají různé rozměry).

</div>

#### Transpozice

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Transpozice)</span></p>

Buď $A \in \mathbb{R}^{m \times n}$. Pak *transponovaná matice* $A^T$ má typ $n \times m$ a je definována $(A^T)_{ij} := a_{ji}$.

Transpozice znamená překlopení dle hlavní diagonály. Díky transpozici můžeme sloupcové vektory $x \in \mathbb{R}^n$ zapisovat jako $x = (x_1, \ldots, x_n)^T$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(Transpozice)</span></p>

Pro číslo $\alpha$ a matice $A, B$ vhodných rozměrů:

1. $(A^T)^T = A$,
2. $(A + B)^T = A^T + B^T$,
3. $(\alpha A)^T = \alpha A^T$,
4. $(AB)^T = B^T A^T$.

Vlastnost (4) lze matematickou indukcí rozšířit na součin $k$ matic: $(A_1 A_2 \ldots A_k)^T = A_k^T \ldots A_2^T A_1^T$.

</div>

#### Symetrická matice

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Symetrická matice)</span></p>

Matice $A \in \mathbb{R}^{n \times n}$ je *symetrická*, pokud $A = A^T$.

</div>

Symetrické matice jsou invariantní vůči transpozici — vizuálně jsou symetrické dle hlavní diagonály. Příkladem symetrických matic je $I_n$, nulová matice $0_n$, nebo $\bigl(\begin{smallmatrix} 1 & 2 \\ 2 & 3 \end{smallmatrix}\bigr)$. Součet symetrických matic je opět symetrická matice, ale pro součin to obecně neplatí.

Pro libovolnou matici $B \in \mathbb{R}^{m \times n}$ je matice $B^T B$ symetrická, neboť $(B^T B)^T = B^T (B^T)^T = B^T B$.

Symetrické matice se často vyskytují v geometrických úlohách (matice vzdáleností), statistice (kovarianční matice) nebo optimalizaci (Hessián).

#### Speciální typy matic

- **Diagonální matice**: $A \in \mathbb{R}^{n \times n}$ je diagonální, pokud $a_{ij} = 0$ pro všechna $i \neq j$. Diagonální matici s prvky $v_1, \ldots, v_n$ na diagonále značíme $\operatorname{diag}(v_1, \ldots, v_n)$.

- **Horní trojúhelníková matice**: $A \in \mathbb{R}^{m \times n}$ je horní trojúhelníková, pokud $a_{ij} = 0$ pro všechna $i > j$. Příkladem je jakákoli matice v REF tvaru (pivoty musí být na nebo nad diagonálou).

- **Dolní trojúhelníková matice**: matice s nulami nad diagonálou.

Součin dvou horních trojúhelníkových matic je opět horní trojúhelníková matice.

#### Skalární součin a vnější součin vektorů

Transpozice a součin vektorů jakožto matic o jednom sloupci umožňují zavést dva důležité součiny:

**Standardní skalární součin** vektorů $x, y \in \mathbb{R}^n$:

$$x^T y = \sum_{i=1}^{n} x_i y_i$$

(formálně matice $1 \times 1$, ztotožníme ji s reálným číslem). Standardní eukleidovská norma:

$$\lVert x \rVert = \sqrt{x^T x} = \sqrt{\sum_{i=1}^{n} x_i^2}.$$

**Vnější součin** vektorů $x \in \mathbb{R}^n$, $y \in \mathbb{R}^n$ je čtvercová matice řádu $n$:

$$xy^T = \begin{pmatrix} x_1 y_1 & x_1 y_2 & \ldots & x_1 y_n \\ x_2 y_1 & x_2 y_2 & \ldots & x_2 y_n \\ \vdots & \vdots & & \vdots \\ x_n y_1 & x_n y_2 & \ldots & x_n y_n \end{pmatrix}.$$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Tvrzení 3.17 — Matice hodnosti 1)</span></p>

Matice $A \in \mathbb{R}^{m \times n}$ má hodnost 1 právě tehdy, když je tvaru $A = xy^T$ pro nějaké nenulové vektory $x \in \mathbb{R}^m$, $y \in \mathbb{R}^n$.

</div>

#### Vlastnosti násobení matice vektorem

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(Tvrzení 3.18)</span></p>

Buď $A \in \mathbb{R}^{m \times n}$, $B \in \mathbb{R}^{n \times p}$, $x \in \mathbb{R}^n$ a $y \in \mathbb{R}^m$. Pak:

1. $Ae_j = A_{*j}$ (násobení jednotkovým vektorem dá $j$-tý sloupec),
2. $e_i^T A = A_{i*}$ (dá $i$-tý řádek),
3. $(AB)_{*j} = A B_{*j}$ ($j$-tý sloupec součinu je $A$ krát $j$-tý sloupec $B$),
4. $(AB)_{i*} = A_{i*} B$ ($i$-tý řádek součinu je $i$-tý řádek $A$ krát $B$),
5. $Ax = \sum_{j=1}^{n} x_j A_{*j}$ (sloupcová interpretace),
6. $y^T A = \sum_{i=1}^{m} y_i A_{i*}$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Zápis soustavy rovnic a interpretace)</span></p>

Soustavu lineárních rovnic můžeme maticově zapsat jako $Ax = b$, kde $x = (x_1, \ldots, x_n)^T$ je vektor proměnných a $b \in \mathbb{R}^m$ vektor pravých stran.

- **Řádková interpretace**: $i$-tá rovnice má tvar $A_{i*} x = b_i$ a popisuje nadrovinu. Hledáme průnik všech nadrovin.
- **Sloupcová interpretace** (z vlastnosti 5): $Ax = b$ znamená $x_1 A_{*1} + x_2 A_{*2} + \ldots + x_n A_{*n} = b$. Hledáme, jak vyjádřit vektor $b$ jako kombinaci sloupců matice $A$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Matice jako zobrazení)</span></p>

Matici $A \in \mathbb{R}^{m \times n}$ lze chápat jako zobrazení $x \mapsto Ax$ z $\mathbb{R}^n$ do $\mathbb{R}^m$. Řešit soustavu $Ax = b$ pak znamená najít všechny vektory $x$, které se zobrazí na vektor $b$.

Složení dvou zobrazení $x \mapsto Ax$ a $y \mapsto By$ odpovídá násobení matic: $x \mapsto B(Ax) = (BA)x$. Proto je maticové násobení definováno právě tímto způsobem.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Matice jako zobrazení v rovině)</span></p>

- **Překlopení podle osy $x_2$**: $A = \bigl(\begin{smallmatrix} -1 & 0 \\ 0 & 1 \end{smallmatrix}\bigr)$, zobrazení $(x_1, x_2)^T \mapsto (-x_1, x_2)^T$.
- **Roztáhnutí ve směru $x_1$**: $A = \bigl(\begin{smallmatrix} 2.5 & 0 \\ 0 & 1 \end{smallmatrix}\bigr)$, zobrazení $(x_1, x_2)^T \mapsto (2.5 x_1, x_2)^T$.
- **Otočení o úhel $\alpha$**: $A = \bigl(\begin{smallmatrix} \cos\alpha & -\sin\alpha \\ \sin\alpha & \cos\alpha \end{smallmatrix}\bigr)$.

Skládání zobrazení odpovídá násobení matic, ale obecně závisí na pořadí (násobení matic není komutativní).

</div>

#### Blokové násobení a výpočetní složitost

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Blokové násobení matic)</span></p>

Matice lze rozdělit do bloků (podmatic) a pak se násobí jako by podmatice byly obyčejná čísla:

$$AB = \begin{pmatrix} A_{11} & A_{12} \\ A_{21} & A_{22} \end{pmatrix} \begin{pmatrix} B_{11} & B_{12} \\ B_{21} & B_{22} \end{pmatrix} = \begin{pmatrix} A_{11}B_{11} + A_{12}B_{21} & A_{11}B_{12} + A_{12}B_{22} \\ A_{21}B_{11} + A_{22}B_{21} & A_{21}B_{12} + A_{22}B_{22} \end{pmatrix}.$$

Je nutné, aby podmatice měly vhodné rozměry, aby součiny a součty v pravé části dávaly smysl.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Výpočetní složitost maticových operací)</span></p>

- **Součet** $A + B$: $n^2$ operací.
- **Součin** $AB$ pro $A, B \in \mathbb{R}^{n \times n}$: celkem $2n^3$ aritmetických operací (standardní postup).

Strassenův algoritmus (1969) snižuje složitost na $\approx n^{2.807}$ operací. Coppersmith–Winograd (1990) dále na $\approx n^{2.376}$. Tyto algoritmy se ale uplatní pouze pro velké $n$. Nejmenší možná asymptotická složitost násobení matic je stále otevřený problém.

</div>

### Regulární matice

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Regulární matice)</span></p>

Buď $A \in \mathbb{R}^{n \times n}$. Matice $A$ je *regulární*, pokud soustava $Ax = 0$ má jediné řešení $x = 0$. V opačném případě se matice $A$ nazývá *singulární*.

Soustava $Ax = 0$ s nulovou pravou stranou se nazývá *homogenní*. Nulový vektor je vždy jejím řešením. Pro $A$ regulární ale žádné jiné řešení neexistuje, tj. $Ax \neq 0$ pro všechna $x \neq 0$. Typickým příkladem regulární matice je $I_n$ a singulární matice $0_n$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Tvrzení 3.27 — Ekvivalentní charakterizace regularity)</span></p>

Buď $A \in \mathbb{R}^{n \times n}$. Pak následující jsou ekvivalentní:

1. $A$ je regulární,
2. $\operatorname{RREF}(A) = I_n$,
3. $\operatorname{rank}(A) = n$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Tvrzení 3.28 — Regulární matice a řešitelnost)</span></p>

Buď $A \in \mathbb{R}^{n \times n}$. Pak následující jsou ekvivalentní:

1. $A$ je regulární,
2. pro nějaké $b \in \mathbb{R}^n$ má soustava $Ax = b$ jediné řešení,
3. pro **každé** $b \in \mathbb{R}^n$ má soustava $Ax = b$ jediné řešení.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Tvrzení 3.29 — Součin regulárních matic)</span></p>

Buďte $A, B \in \mathbb{R}^{n \times n}$ regulární matice. Pak $AB$ je také regulární.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Důkaz</summary>

Buď $x$ řešení soustavy $ABx = 0$. Chceme ukázat, že $x$ musí být nulový vektor. Označme $y := Bx$. Pak soustava přechází na $Ay = 0$. Z regularity matice $A$ je jediné řešení $y = 0$, což dává $Bx = 0$. Z regularity matice $B$ je pak $x = 0$.

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Tvrzení 3.30 — Singularita součinu)</span></p>

Je-li aspoň jedna z matic $A, B \in \mathbb{R}^{n \times n}$ singulární, pak $AB$ je také singulární.

</div>

#### Matice elementárních úprav

Elementární řádkové úpravy jdou reprezentovat maticově — výsledek úpravy na matici $A$ se dá vyjádřit jako $EA$ pro nějakou matici $E$. Matice $E$ získáme aplikací dané úpravy na jednotkovou matici $I_n$.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Matice elementárních úprav)</span></p>

1. **Vynásobení $i$-tého řádku číslem $\alpha \neq 0$**: $E_i(\alpha) = I + (\alpha - 1)e_i e_i^T$ (na diagonále v $i$-tém řádku je $\alpha$ místo $1$).
2. **Přičtení $\alpha$-násobku $j$-tého řádku k $i$-tému**: $E_{ij}(\alpha) = I + \alpha e_i e_j^T$ ($i \neq j$; na pozici $(i, j)$ je $\alpha$).
3. **Výměna $i$-tého a $j$-tého řádku**: $E_{ij} = I + (e_j - e_i)(e_i - e_j)^T$.

Všechny matice elementárních operací jsou regulární. Každá elementární úprava má svoji inverzní úpravu, čímž se matice převede zpět na jednotkovou.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Věta 3.31 — RREF jako součin)</span></p>

Buď $A \in \mathbb{R}^{m \times n}$. Pak $\operatorname{RREF}(A) = QA$ pro nějakou regulární matici $Q \in \mathbb{R}^{m \times m}$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Tvrzení 3.32 — Regulární matice jako součin elementárních)</span></p>

Každá regulární matice $A \in \mathbb{R}^{n \times n}$ se dá vyjádřit jako součin konečně mnoha elementárních matic.

</div>

### Inverzní matice

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Inverzní matice)</span></p>

Buď $A \in \mathbb{R}^{n \times n}$. Pak $A^{-1}$ je *inverzní maticí* k $A$, pokud splňuje $AA^{-1} = A^{-1}A = I_n$.

Matice inverzní k $I_n$ je opět $I_n$. Inverzní matice k nulové matici $0_n$ evidentně neexistuje.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Věta 3.34 — Existence inverzní matice)</span></p>

Buď $A \in \mathbb{R}^{n \times n}$. Je-li $A$ regulární, pak k ní existuje inverzní matice a je určená jednoznačně. Naopak, existuje-li k $A$ inverzní, pak $A$ musí být regulární.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Důkaz</summary>

**Existence.** Z regularity matice $A$ plyne, že soustava $Ax = e_j$ má jediné řešení pro každé $j = 1, \ldots, n$; označme je $x_j$. Vytvořme matici $A^{-1} = (x_1 \mid x_2 \mid \ldots \mid x_n)$. Pak $(AA^{-1})_{*j} = Ax_j = e_j = I_{*j}$, tedy $AA^{-1} = I$. Druhou rovnost $A^{-1}A = I$ dokážeme trikem: matice $A(A^{-1}A - I)$ je nulová, tedy její $j$-tý sloupec $A(A^{-1}A - I)_{*j} = 0$. Z regularity $A$ plyne $(A^{-1}A - I)_{*j} = 0$ pro každé $j$, tedy $A^{-1}A = I$.

**Jednoznačnost.** Nechť pro nějakou matici $B$ platí $AB = BA = I$. Pak $B = BI = B(AA^{-1}) = (BA)A^{-1} = IA^{-1} = A^{-1}$.

**Naopak.** Nechť pro $A$ existuje inverzní matice. Buď $x$ řešení $Ax = 0$. Pak $x = Ix = (A^{-1}A)x = A^{-1}(Ax) = A^{-1}0 = 0$. Tedy $A$ je regulární.

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Tvrzení 3.35)</span></p>

Je-li $A$ regulární, pak $A^T$ je regulární a $(A^T)^{-1} = (A^{-1})^T$, což se někdy zkráceně značí $A^{-T}$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Věta 3.36 — Jedna rovnost stačí)</span></p>

Buďte $A, B \in \mathbb{R}^{n \times n}$. Je-li $BA = I_n$, pak obě matice $A, B$ jsou regulární a navzájem k sobě inverzní, to jest $B = A^{-1}$ a $A = B^{-1}$.

</div>

#### Výpočet inverzní matice

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Věta 3.37 — Výpočet inverzní matice)</span></p>

Buď $A \in \mathbb{R}^{n \times n}$. Nechť matice $(A \mid I_n)$ typu $n \times 2n$ má RREF tvar $(I_n \mid B)$. Pak $B = A^{-1}$. Netvoří-li první část RREF tvar $I_n$, pak $A$ je singulární.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Výpočet inverzní matice)</span></p>

Buď $A = \begin{pmatrix} 1 & 1 & 3 \\ 0 & 2 & -1 \\ 3 & 5 & 7 \end{pmatrix}$. Inverzní matici spočítáme:

$$(A \mid I_3) = \begin{pmatrix} 1 & 1 & 3 & 1 & 0 & 0 \\ 0 & 2 & -1 & 0 & 1 & 0 \\ 3 & 5 & 7 & 0 & 0 & 1 \end{pmatrix} \xrightarrow{\text{RREF}} \begin{pmatrix} 1 & 0 & 0 & -9.5 & -4 & 3.5 \\ 0 & 1 & 0 & 1.5 & 1 & -0.5 \\ 0 & 0 & 1 & 3 & 1 & -1 \end{pmatrix} = (I_3 \mid A^{-1}).$$

Tedy $A^{-1} = \begin{pmatrix} -9.5 & -4 & 3.5 \\ 1.5 & 1 & -0.5 \\ 3 & 1 & -1 \end{pmatrix}$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(Tvrzení 3.39 — Vlastnosti inverzní matice)</span></p>

Buďte $A, B \in \mathbb{R}^{n \times n}$ regulární. Pak:

1. $(A^{-1})^{-1} = A$,
2. $(A^{-1})^T = (A^T)^{-1}$,
3. $(\alpha A)^{-1} = \frac{1}{\alpha} A^{-1}$ pro $\alpha \neq 0$,
4. $(AB)^{-1} = B^{-1} A^{-1}$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Věta 3.40 — Soustava rovnic a inverzní matice)</span></p>

Buď $A \in \mathbb{R}^{n \times n}$ regulární. Pak řešení soustavy $Ax = b$ je dáno vzorcem $x = A^{-1}b$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Poznámka o praktickém použití)</span></p>

Výpočet $x = A^{-1}b$ v praxi nepoužíváme, neboť je časově dražší než Gaussova eliminace. Význam věty 3.40 je spíše teoretický — udává explicitní vzorec pro řešení.

Přenásobení soustavy $Ax = b$ regulární maticí $Q$ zleva nemění množinu řešení: od soustavy $(QA)x = Qb$ se lze přenásobením $Q^{-1}$ vrátit k původní. Každou regulární matici lze složit z elementárních matic (tvrzení 3.32), takže elementární řádkové úpravy skutečně nemění množinu řešení.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Geometrická interpretace regularity)</span></p>

Zobrazení $x \mapsto Ax$ s regulární maticí $A \in \mathbb{R}^{n \times n}$ je bijekce — každý vektor z $\mathbb{R}^n$ má svůj jediný vzor. Inverzní zobrazení je $y \mapsto A^{-1}y$. Příklady: otočení, převrácení, natažení (regulární). Naopak, singulární matice deformuje prostor — projekce $A = \bigl(\begin{smallmatrix} 1 & 0 \\ 0 & 0 \end{smallmatrix}\bigr)$ zobrazí celou rovinu na osu $x_1$, a proto nemá inverzi.

</div>

#### Shermanova–Morrisonové formule

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Shermanova–Morrisonové formule — Věta 3.44)</span></p>

Buď $A \in \mathbb{R}^{n \times n}$ regulární a mějme $b, c \in \mathbb{R}^n$. Pokud $c^T A^{-1} b = -1$, tak $A + bc^T$ je singulární; jinak

$$(A + bc^T)^{-1} = A^{-1} - \frac{1}{1 + c^T A^{-1} b} A^{-1} b c^T A^{-1}.$$

</div>

Tato formule umožňuje rychle přepočítat inverzní matici, pokud původní matici "málo" změníme (rank-one update). Známe-li $A^{-1}$, potom ke spočítání $(A + bc^T)^{-1}$ potřebujeme pouze $6n^2$ operací místo $\sim n^3$.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Výpočetní složitost inverze)</span></p>

Výpočetní složitost inverze matice $A \in \mathbb{R}^{n \times n}$ je dána složitostí algoritmu RREF na matici $(A \mid I_n)$, tedy řádově $3n^3$ operací. Ve skutečnosti lze postup vylepšit, aby celková složitost byla $2n^3$, tedy stejná asymptotická složitost jako součin matic.

</div>

### LU rozklad

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(LU rozklad)</span></p>

*LU rozklad* čtvercové matice $A \in \mathbb{R}^{n \times n}$ je rozklad na součin $A = LU$, kde $L$ je dolní trojúhelníková matice s jedničkami na diagonále a $U$ horní trojúhelníková matice.

</div>

LU rozklad úzce souvisí s odstupňovaným tvarem matice. Za matici $U$ můžeme vzít odstupňovaný tvar matice $A$ a matici $L$ získáme z elementárních úprav. Pokud při eliminaci používáme pouze přičtení násobku řádku k řádku pod ním (bez prohazování řádků), tak matice takovýchto úprav $E_{ij}(\alpha)$ jsou dolní trojúhelníkové a jejich součin $L$ je opět dolní trojúhelníková matice.

Převeď $A$ na odstupňovaný tvar $U$: $E_k \ldots E_1 A = U$, z čehož $A = \underbrace{E_1^{-1} \ldots E_k^{-1}}_{L} U$.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(LU rozklad)</span></p>

$$A = \begin{pmatrix} 2 & 1 & 3 \\ 4 & 1 & 7 \\ -6 & -2 & -12 \end{pmatrix} = \begin{pmatrix} 1 & 0 & 0 \\ 2 & 1 & 0 \\ -3 & -1 & 1 \end{pmatrix} \begin{pmatrix} 2 & 1 & 3 \\ 0 & -1 & 1 \\ 0 & 0 & -2 \end{pmatrix} = LU.$$

Matici $L$ konstruujeme tak, že namísto nul pod diagonálou budeme zapisovat koeficienty $-\alpha$ z elementárních úprav.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Použití LU rozkladu pro řešení soustavy)</span></p>

Pro řešení soustavy $Ax = b$ (tedy $LUx = b$):

1. Najdi LU rozklad matice $A$, tj. $A = LU$.
2. Vyřeš soustavu $Ly = b$ dopřednou substitucí.
3. Vyřeš soustavu $Ux = y$ zpětnou substitucí.

Výpočetní složitost celého algoritmu je asymptoticky stejná jako u Gaussovy eliminace ($\frac{2}{3}n^3$).

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(Důsledek 3.49 — PA = LU)</span></p>

Každá matice $A \in \mathbb{R}^{n \times n}$ jde rozložit na tvar $PA = LU$, kde $P \in \mathbb{R}^{n \times n}$ je *permutační matice* (matice s jedničkami na diagonále po vhodném přeuspořádání řádků), $L \in \mathbb{R}^{n \times n}$ je dolní trojúhelníková matice s jedničkami na diagonále a $U \in \mathbb{R}^{n \times n}$ horní trojúhelníková matice.

</div>

LU rozklad bez prohazování řádků nemusí pro každou matici existovat (např. pro $A = \bigl(\begin{smallmatrix} 0 & 1 \\ 1 & 0 \end{smallmatrix}\bigr)$), ale po vhodném prohození řádků (permutační matice $P$) už ano.

### Numerická stabilita při řešení soustav, iterativní metody

#### Špatně podmíněné matice

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Numerická stabilita)</span></p>

Při numerickém řešení soustav lineárních rovnic na počítačích dochází k zaokrouhlovacím chybám a vypočtený výsledek se může diametrálně lišit od správného řešení. Chyby se projevují zejména u tzv. **špatně podmíněných matic** — matic, které jsou v jistém smyslu blízko singulárním maticím.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Špatně podmíněná soustava)</span></p>

Dvě soustavy lišící se v jediném koeficientu o $\frac{2}{30}$:

$$0.835 x_1 + 0.667 x_2 = 0.168, \quad 0.333 x_1 + 0.266 x_2 = 0.067$$

má řešení $(1, -1)$, zatímco

$$0.835 x_1 + 0.667 x_2 = 0.168, \quad 0.333 x_1 + 0.266 x_2 = 0.066$$

má řešení $(-666, 834)$. Geometricky jde o průsečík dvou téměř rovnoběžných přímek — malá změna v datech vede k obrovské změně průsečíku.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Hilbertova matice)</span></p>

Typickým příkladem špatně podmíněné matice je Hilbertova matice $H_n$ řádu $n$, definovaná $(H_n)_{ij} = \frac{1}{i + j - 1}$. Již při $n \approx 14$ mají numerické chyby enormní vliv na přesnost řešení (při double precision $\approx 10^{-16}$).

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Parciální a úplná pivotizace)</span></p>

**Parciální pivotizace** (volba pivota s maximální absolutní hodnotou ve sloupci pod aktuální pozicí) často vede k přesnějšímu řešení, i když ani ta není všelék. **Úplná pivotizace** hledá prvek s maximální absolutní hodnotou v celé podmatici vpravo dole — zlepšuje numerické vlastnosti, ale je výpočetně náročnější a v praxi se moc nepoužívá.

</div>

#### Iterativní metody

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Řídké soustavy a iterativní metody)</span></p>

Velké praktické úlohy (typicky řešení diferenciálních rovnic) vedou na velké, ale **řídké** soustavy $Ax = b$ (řád matice $n = 10^7$, ale většina prvků je nulová). Gaussova eliminace není vhodná, protože elementární úpravy zvyšují podíl nenulových prvků ("zahušťují" matici) a navíc eliminace nevyužívá řídkost.

**Iterativní metody** od počátečního vektoru postupně konvergují k řešení. Jsou méně citlivé na zaokrouhlovací chyby, mají menší paměťové nároky a hodí se pro řídké matice. Jedna iterace vyžaduje řádově $kn$ operací (kde $k$ je počet nenulových prvků v řádku), zatímco Gaussova eliminace vyžaduje $\frac{2}{3}n^3$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Gaussova–Seidelova metoda)</span></p>

Uvažujme soustavu:

$$6x + 2y - z = 4, \quad x + 5y + z = 3, \quad 2x + y + 4z = 27.$$

Přepíšeme: $x = \frac{1}{6}(4 - 2y + z)$, $y = \frac{1}{5}(3 - x - z)$, $z = \frac{1}{4}(27 - 2x - y)$.

S počátečním odhadem $(1, 1, 1)$ iterujeme a po 6 iteracích máme přibližné řešení blízké skutečnému $(2, -1, 6)^T$:

| iterace | $x$ | $y$ | $z$ |
| --- | --- | --- | --- |
| 0 | 1 | 1 | 1 |
| 1 | 0.5 | 0.3 | 6.425 |
| 2 | 1.6375 | $-1.0125$ | 6.184375 |
| 6 | 1.999624 | $-0.999895$ | 6.000012 |

Konvergence je zaručena jen pro určité třídy matic, např. když v každém řádku je diagonální prvek větší než součet absolutních hodnot zbývajících prvků.

</div>

### Aplikace

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Leontiefův model ekonomiky)</span></p>

Uvažujme ekonomiku s $n$ sektory. Sektor $i$ vyrábí jednu komoditu o množství $x_i$. Výroba jednotky $j$-té komodity potřebuje $a_{ij}$ jednotek $i$-té komodity. Označme $d_i$ výsledný požadavek na výrobu sektoru $i$. Model:

$$x_i = a_{i1} x_1 + \ldots + a_{in} x_n + d_i, \quad \text{neboli} \quad (I_n - A)x = d.$$

Řešení má explicitní vyjádření $x = (I_n - A)^{-1} d$. Leontief aplikoval model na ekonomiku USA ve 40. letech 20. století a za tento vstupně-výstupní model obdržel roku 1973 Nobelovu cenu.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Interpolace polynomem)</span></p>

Mějme $n + 1$ bodů $(x_0, y_0), (x_1, y_1), \ldots, (x_n, y_n)$ v rovině, kde $x_i \neq x_j$ pro $i \neq j$. Hledáme polynom $p(x) = a_n x^n + \ldots + a_1 x + a_0$ procházející těmito body. Dosazením dostáváme soustavu rovnic s **Vandermondovou maticí**:

$$\begin{pmatrix} x_0^n & \ldots & x_0 & 1 \\ x_1^n & \ldots & x_1 & 1 \\ \vdots & & \vdots & \vdots \\ x_n^n & \ldots & x_n & 1 \end{pmatrix} \begin{pmatrix} a_n \\ \vdots \\ a_1 \\ a_0 \end{pmatrix} = \begin{pmatrix} y_0 \\ y_1 \\ \vdots \\ y_n \end{pmatrix}.$$

Vandermondova matice je regulární (pro různá $x_i$), a proto polynom stupně $n$ vždy existuje a je určený jednoznačně.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Diskrétní a rychlá Fourierova transformace)</span></p>

Polynom $p(x)$ lze reprezentovat dvěma způsoby: (1) koeficienty $a_n, \ldots, a_0$ nebo (2) seznamem funkčních hodnot v $n + 1$ různých bodech. V první reprezentaci je sčítání jednoduché ($\sim n$ operací), ale násobení stojí $\sim 2n^2$. Ve druhé je sčítání i násobení $\sim n$ operací.

**Rychlá Fourierova transformace** (FFT) umožňuje přecházet mezi oběma reprezentacemi v $\sim \alpha n \log(n)$ operacích, a tím pádem i násobit polynomy (a tedy i reálná čísla) efektivně.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Komprese obrázku — Haarova transformace)</span></p>

Obrázek reprezentovaný maticí $M \in \mathbb{R}^{m \times n}$ lze komprimovat pomocí Haarovy transformace. Matici $M$ rozdělíme na podmatice $8 \times 8$ a na každou aplikujeme transformaci $A' = H^T A H$, kde $H$ je regulární matice. Průměrováním sousedních pixelů se hodnoty blízké nule vynulují — tím vznikne řídká matice, kterou stačí ukládat efektivněji. *Kompresní poměr* $k$ udává poměr nenulových čísel v $A'$ před a po vynulování malých hodnot. Vyšší $k$ znamená vyšší kompresi, ale i vyšší ztrátu informace.

</div>

## Kapitola 4 — Grupy a tělesa

Tato kapitola je věnovaná základním algebraickým strukturám — grupám a tělesům. Jsou to abstraktní pojmy zobecňující dobře známé obory reálných (racionálních, komplexních aj.) čísel s operacemi sčítání a násobení.

### 4.1 Grupy

Grupa je velmi abstraktní algebraická struktura. Jedná se o množinu s binární operací, která musí splňovat několik základních vlastností. Pomocí grup se popisují symetrie (nejen geometrických) objektů. Díky jejich obecnosti a abstraktnosti můžeme grupy najít mnoha v různých oborech: fyzika (Lieovy grupy), architektura (Friezovy grupy), geometrie a molekulární chemie (symetrické grupy) aj.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Definice 4.1 — Grupa)</span></p>

Buď $\circ \colon G^2 \to G$ binární operace na množině $G$. Pak *grupa* je dvojice $(G, \circ)$ splňující:

1. $\forall a, b, c \in G: a \circ (b \circ c) = (a \circ b) \circ c$ &emsp; (asociativita),
2. $\exists e \in G\ \forall a \in G: e \circ a = a \circ e = a$ &emsp; (existence neutrálního prvku),
3. $\forall a \in G\ \exists b \in G: a \circ b = b \circ a = e$ &emsp; (existence inverzního prvku).

**Abelova (komutativní) grupa** je taková grupa, která navíc splňuje:

4. $\forall a, b \in G: a \circ b = b \circ a$ &emsp; (komutativita).

</div>

V definici grupy je implicitně schována podmínka uzavřenosti, aby výsledek operace nevypadl ven z množiny $G$. Pokud je operací $\circ$ sčítání, většinou se značí neutrální prvek $0$ a inverzní $-a$; pokud jde o násobení, neutrální prvek se označuje $1$ a inverzní $a^{-1}$.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Poznámka 4.2 — Definice konstrukcí vs. axiomy)</span></p>

Matematický objekt lze zavést buď konstrukcí z nějakých již vytvořených objektů, nebo specifikací vlastností (axiomů), které má splňovat. Definice grupy spadá do druhé skupiny. Grupou pak je jakýkoli objekt, který splňuje dané vlastnosti. Axiomatická definice má tu výhodu, že nás nesvazuje s jedním konkrétním objektem — jakoukoli vlastnost, kterou odvodíme pro axiomaticky definovaný objekt, potom automaticky platí pro každý konkrétní případ.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 4.3 — Příklady grup)</span></p>

- Dobře známé obory celých čísel $(\mathbb{Z}, +)$, racionálních čísel $(\mathbb{Q}, +)$, reálných čísel $(\mathbb{R}, +)$ a komplexních čísel $(\mathbb{C}, +)$. Neutrálním prvkem je $0$, inverzním prvkem k prvku $a$ je $-a$. Komutativita a asociativita sčítání zjevně platí.
- Grupy matic $(\mathbb{R}^{m \times n}, +)$. Neutrálním prvkem je nulová matice $0$ rozměru $m \times n$, inverzním prvkem k matici $A$ je $-A$. Komutativita a asociativita sčítání platí s ohledem na tvrzení 3.5.
- Konečná grupa $(\mathbb{Z}_n, +)$, kde množina $\mathbb{Z}_n := \lbrace 0, 1, \ldots, n - 1 \rbrace$ a sčítání se provádí modulo $n$. Neutrálním prvkem je $0$, inverzním prvkem k prvku $a$ je $-a \bmod n$.
- Číselné obory s násobením, např. $(\mathbb{Q} \setminus \lbrace 0 \rbrace, \cdot)$, $(\mathbb{R} \setminus \lbrace 0 \rbrace, \cdot)$. Nulu musíme vynechat, protože nemá inverzní prvek. Neutrálním prvkem je nyní $1$, inverzním prvkem k prvku $a$ je $a^{-1}$.
- Množina reálných polynomů proměnné $x$ se sčítáním.
- Zobrazení na množině s operací skládání, např. rotace v $\mathbb{R}^n$ podle počátku nebo později probírané permutace (sekce 4.2). Rotace v rovině $\mathbb{R}^2$ jsou ještě komutativní, ale ve vyšších dimenzích komutativitu ztrácíme. Neutrálním prvkem je otočení o nulový úhel, inverzním prvkem je otočení o opačný úhel zpět.
- Regulární matice pevného řádu $n$ s násobením (tzv. maticová grupa). Neutrálním prvkem je $I_n$, inverzním prvkem k matici $A$ je inverzní matice $A^{-1}$. Asociativita maticového součinu byla nahlédnuta ve tvrzení 3.9.

**Příklady negrup:** $(\mathbb{N}, +)$, $(\mathbb{Z}, -)$, $(\mathbb{R} \setminus \lbrace 0 \rbrace, :)$, ...

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Tvrzení 4.4 — Základní vlastnosti v grupě)</span></p>

Pro prvky grupy $(G, \circ)$ platí následující vlastnosti:

1. $a \circ c = b \circ c$ implikuje $a = b$ &emsp; (tzv. krácení),
2. neutrální prvek $e$ je určen jednoznačně,
3. pro každé $a \in G$ je jeho inverzní prvek určen jednoznačně,
4. rovnice $a \circ x = b$ má právě jedno řešení pro každé $a, b \in G$,
5. $(a^{-1})^{-1} = a$,
6. $(a \circ b)^{-1} = b^{-1} \circ a^{-1}$.

*Důkaz.* (1) Z $a \circ c = b \circ c$ složíme zprava $c^{-1}$: $a \circ (c \circ c^{-1}) = b \circ (c \circ c^{-1})$, tedy $a \circ e = b \circ e$, tj. $a = b$. (2) Existují-li dva různé neutrální prvky $e_1, e_2$, pak $e_1 = e_1 \circ e_2 = e_2$, spor. (3) Existují-li dva různé inverzní prvky $a_1, a_2$, pak $a \circ a_1 = e = a \circ a_2$ a z vlastnosti krácení $a_1 = a_2$, spor. (4) Vynásobíme $a \circ x = b$ zleva prvkem $a^{-1}$ a dostaneme $x = a^{-1} \circ b$. Dosazením ověříme, že rovnost splňuje.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Definice 4.5 — Podgrupa)</span></p>

*Podgrupa* grupy $(G, \circ)$ je grupa $(H, \diamond)$ taková, že $H \subseteq G$ a pro všechna $a, b \in H$ platí $a \circ b = a \diamond b$. Značení: $(H, \circ) \le (G, \circ)$.

Jinými slovy, se stejně definovanou operací splňuje $H$ vlastnosti uzavřenost a existence neutrálního a inverzního prvku. To jest, pro každé $a, b \in H$ je $a \circ b \in H$, dále $e \in H$, a pro každé $a \in H$ je $a^{-1} \in H$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 4.6)</span></p>

- Každá grupa $(G, \circ)$ má dvě triviální podgrupy: sama sebe $(G, \circ)$ a $(\lbrace e \rbrace, \circ)$.
- $(\mathbb{N}, +) \not\le (\mathbb{Z}, +) \le (\mathbb{Q}, +) \le (\mathbb{R}, +) \le (\mathbb{C}, +)$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 4.7)</span></p>

Podgrupy jsou uzavřené na průnik, ale ne na sjednocení. Jinými slovy, průnik dvou podgrup grupy $(G, \circ)$ je opět její podgrupa, ale sjednocení podgrup již podgrupa není.

</div>

### 4.2 Permutace

Dalším příkladem grup je takzvaná symetrická grupa permutací, proto si povíme něco více o permutacích.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Definice 4.8 — Permutace)</span></p>

*Permutace* na konečné množině $X$ je vzájemně jednoznačné zobrazení $p \colon X \to X$.

</div>

Většinou budeme uvažovat $X = \lbrace 1, \ldots, n \rbrace$. Množina všech permutací na množině $\lbrace 1, \ldots, n \rbrace$ se pak značí $S_n$. Zadání permutace je možné například:

- **Tabulkou**, kde nahoře jsou vzory a dole jejich obrazy.
- **Grafem** vyznačujícím kam se který prvek zobrazí.
- **Rozložením na cykly**: $p = (1, 2)(3)(4, 5, 6)$, kde každá závorka $(a_1, \ldots, a_k)$ znamená, že $a_1$ se zobrazí na $a_2$, $a_2$ se zobrazí na $a_3$, atd. až $a_{k-1}$ se zobrazí na $a_k$ a $a_k$ se zobrazí na $a_1$. Z definice je patrné, že každou permutaci lze rozložit na disjunktní cykly. V následujícím textu budeme nejčastěji používat redukovaný zápis, ve kterém vynecháváme cykly délky $1$.

Příkladem jednoduché, ale netriviální, permutace je *transpozice* $t = (i, j)$ — permutace s jedním cyklem délky $2$ prohazující dva prvky. Nejjednodušší je identita $id$ zobrazující každý prvek na sebe.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Definice 4.9 — Inverzní permutace)</span></p>

Buď $p \in S_n$. *Inverzní permutace* k $p$ je permutace $p^{-1}$ definovaná $p^{-1}(i) = j$, pokud $p(j) = i$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 4.10)</span></p>

$(i, j)^{-1} = (i, j)$, $(i, j, k)^{-1} = (k, j, i)$, ...

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Definice 4.11 — Skládání permutací)</span></p>

Buďte $p, q \in S_n$. *Složená permutace* $p \circ q$ je permutace definovaná $(p \circ q)(i) = p(q(i))$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 4.12)</span></p>

$id \circ p = p \circ id = p$, $p \circ p^{-1} = p^{-1} \circ p = id$, ...

</div>

Skládání permutací je asociativní (jako každé zobrazení), ale komutativní obecně není. Například pro $p = (1, 2)$, $q = (1, 3, 2)$ máme $p \circ q = (1, 3)$, ale $q \circ p = (2, 3)$.

#### Znaménko permutace

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Definice 4.13 — Znaménko permutace)</span></p>

Nechť se permutace $p \in S_n$ skládá z $k$ cyklů. Pak *znaménko permutace* je číslo $\operatorname{sgn}(p) = (-1)^{n-k}$.

</div>

Znaménko je vždy $1$ nebo $-1$. Podle toho se také rozdělují permutace na *sudé* (ty, co mají znaménko $1$) a na *liché* (ty se znaménkem $-1$).

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Věta 4.15 — O znaménku složení permutace a transpozice)</span></p>

Buď $p \in S_n$ a buď $t = (i, j)$ transpozice. Pak $\operatorname{sgn}(p) = -\operatorname{sgn}(t \circ p) = -\operatorname{sgn}(p \circ t)$.

*Důkaz.* Dokážeme $\operatorname{sgn}(p) = -\operatorname{sgn}(t \circ p)$, druhá rovnost je analogická. Permutace $p$ se skládá z několika cyklů. Rozlišme dva případy:

- Nechť $i, j$ jsou částí stejného cyklu, označme jej $(i, u_1, \ldots, u_r, j, v_1, \ldots, v_s)$. Pak $(i, j) \circ (i, u_1, \ldots, u_r, j, v_1, \ldots, v_s) = (i, u_1, \ldots, u_r)(j, v_1, \ldots, v_s)$, tedy počet cyklů se zvýší o jedna.
- Nechť $i, j$ náleží do dvou různých cyklů, např. $(i, u_1, \ldots, u_r)(j, v_1, \ldots, v_s)$. Pak $(i, j) \circ (i, u_1, \ldots, u_r)(j, v_1, \ldots, v_s) = (i, u_1, \ldots, u_r, j, v_1, \ldots, v_s)$, tedy počet cyklů se sníží o jedna.

V každém případě se počet cyklů změní o jedna, a tudíž i výsledné znaménko.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Věta 4.16 — Rozklad na transpozice)</span></p>

Každou permutaci lze rozložit na složení transpozic. Libovolný cyklus $(u_1, \ldots, u_r)$ se rozloží

$$(u_1, \ldots, u_r) = (u_1, u_2) \circ (u_2, u_3) \circ (u_3, u_4) \circ \ldots \circ (u_{r-1}, u_r).$$

Rozklad na transpozice není jednoznačný, dokonce ani počet transpozic ne. Pouze jejich parita zůstane stejná.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(Důsledek 4.17)</span></p>

Platí $\operatorname{sgn}(p) = (-1)^r$, kde $r$ je počet transpozic při rozkladu $p$ na transpozice.

*Důkaz.* Je to důsledek věty 4.15. Vyjdeme z identity, která je sudá. Každá transpozice mění znaménko, tedy výsledné znaménko bude $(-1)^r$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(Důsledek 4.18)</span></p>

Buď $p, q \in S_n$. Pak $\operatorname{sgn}(p \circ q) = \operatorname{sgn}(p) \operatorname{sgn}(q)$.

*Důkaz.* Nechť $p$ se dá rozložit na $r_1$ transpozic a $q$ na $r_2$ transpozic. Tedy $p \circ q$ lze složit z $r_1 + r_2$ transpozic. Pak $\operatorname{sgn}(p \circ q) = (-1)^{r_1 + r_2} = (-1)^{r_1}(-1)^{r_2} = \operatorname{sgn}(p)\operatorname{sgn}(q)$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(Důsledek 4.19)</span></p>

Buď $p \in S_n$. Pak $\operatorname{sgn}(p) = \operatorname{sgn}(p^{-1})$.

*Důkaz.* Platí $1 = \operatorname{sgn}(id) = \operatorname{sgn}(p \circ p^{-1}) = \operatorname{sgn}(p)\operatorname{sgn}(p^{-1})$, tedy $p, p^{-1}$ musí mít stejné znaménko.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Poznámka 4.20 — Inverze a znaménko)</span></p>

Kromě počtu cyklů a počtu transpozic jde znaménko permutace $p$ zavést také například pomocí počtu inverzí. Inverzí zde rozumíme uspořádanou dvojici $(i, j)$ takovou, že $i < j$ a $p(i) > p(j)$. Označíme-li počet inverzí permutace $p$ jako $I(p)$, pak platí $\operatorname{sgn}(p) = (-1)^{I(p)}$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Poznámka 4.21 — Symetrická grupa)</span></p>

Množina permutací $S_n$ s operací skládání $\circ$ tvoří nekomutativní grupu $(S_n, \circ)$, tzv. *symetrickou grupu*. Dá se ukázat, že každá grupa je isomorfní nějaké podgrupě symetrické grupy (tzv. Cayleyova reprezentace, dokonce platí i zobecnění na nekonečné grupy). Podobnou roli hrají maticové grupy, protože každá konečná grupa je isomorfní nějaké maticové podgrupě (lineární reprezentace).

Grupa $(S_n, \circ)$ se nazývá symetrická, protože ona a její podgrupy popisují symetrie různých objektů. Kupříkladu rovnoramenný trojúhelník je symetrický podle svislé osy, a této symetrii odpovídá permutace $(2, 3)$. Symetrie rovnostranného trojúhelníku jsou souměrnosti podle těžnic (odpovídají transpozicím $(1, 2)$, $(2, 3)$ a $(1, 3)$) a dále otočení o $0°$, $120°$ a o $240°$ (odpovídají permutacím $id$, $(1, 2, 3)$ a $(1, 3, 2)$). Všechny symetrie tedy představují celou grupu $(S_3, \circ)$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 4.22 — Loydova patnáctka)</span></p>

Symetrické grupy a znaménko permutace se využijí také při analýze hlavolamů jako je Loydova patnáctka nebo Rubikova kostka. Loydova patnáctka je hra, která sestává z pole $4 \times 4$ a z kachlíků očíslovaných $1$ až $15$. Jedno pole je prázdné a přesouváním sousedních kachlíků na prázdné pole měníme rozložení kachlíků. Cílem je dospět pomocí těchto přesunů k vzestupnému uspořádání kachlíků.

Jestliže očíslujeme jednotlivá políčka jako $1$ až $16$, pak konfigurace kachlíků odpovídá nějaké permutaci $p \in S_{16}$ a přesun kachlíku odpovídá složení $p$ s nějakou transpozicí. Označíme-li $(r, s)$ pozici prázdného pole, pak hodnota $h = (-1)^{r+s}\operatorname{sgn}(p)$ zůstává po celou hru stejná. Cílová konfigurace má hodnotu $h = 1$, tedy počáteční konfigurace s $h = -1$ řešitelné být nemohou.

</div>

### 4.3 Tělesa

Algebraická tělesa zobecňují třídu tradičních číselných oborů jako je třeba množina reálných čísel na abstraktní množinu se dvěma operacemi a řadou vlastností. To nám umožní pracovat s maticemi (sčítat, násobit, invertovat, řešit soustavy rovnic, ...) nad jinými obory než jen nad $\mathbb{R}$.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 4.23 — Motivační)</span></p>

Uvažujme soustavu lineárních rovnic $Ax = b$ s regulární maticí $A$. Soustava má tudíž jediné řešení. Jsou-li prvky matice $(A \mid b)$ celá čísla, pak řešení nemusí mít celočíselné složky, protože během úprav dochází k dělení. Jsou-li ale prvky matice $(A \mid b)$ racionální čísla, pak řešení má také racionální složky, protože běžnými maticovými úpravami provádíme pouze aritmetické operace s čísly. Množina racionálních čísel $\mathbb{Q}$ má tedy tu vlastnost, že běžnými maticovými operacemi se nedostaneme mimo $\mathbb{Q}$. Podobnou vlastnost má také například množina reálných čísel $\mathbb{R}$ a množina komplexních čísel $\mathbb{C}$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Definice 4.24 — Těleso)</span></p>

*Těleso* je množina $\mathbb{T}$ spolu se dvěma komutativními binárními operacemi $+$ a $\cdot$ splňující

1. $(\mathbb{T}, +)$ je Abelova grupa, neutrální prvek značíme $0$ a inverzní k $a$ pak $-a$,
2. $(\mathbb{T} \setminus \lbrace 0 \rbrace, \cdot)$ je Abelova grupa, neutrální prvek značíme $1$ a inverzní k $a$ pak $a^{-1}$,
3. $\forall a, b, c \in \mathbb{T}: a \cdot (b + c) = a \cdot b + a \cdot c$ &emsp; (distributivita).

Každé těleso má aspoň dva prvky, protože z definice nutně vyplývá, že $0 \neq 1$. Operace $+$ a $\cdot$ nemusí nutně představovat klasické sčítání a násobení, ale toto značení se používá pro korespondenci se standardními číselnými obory. Z tohoto důvodu budeme také zkráceně psát $ab$ namísto $a \cdot b$.

</div>

Vlastnost inverzního prvku v grupě $(\mathbb{T}, +)$ přirozeně zavádí operaci „$-$" definovanou jako přičtení inverzního prvku, tj. $a - b \equiv a + (-b)$. Analogicky vlastnost inverzního prvku v grupě $(\mathbb{T} \setminus \lbrace 0 \rbrace, \cdot)$ přirozeně zavádí operaci „$/$" definovanou jako násobení inverzním prvkem, tj. $a / b \equiv ab^{-1}$.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 4.25)</span></p>

Příkladem nekonečných těles je například $\mathbb{Q}$, $\mathbb{R}$ a $\mathbb{C}$ s běžnými operacemi sčítání a násobení. Množina celých čísel $\mathbb{Z}$ ale těleso netvoří, protože chybí inverzní prvky pro násobení (např. když invertujeme celočíselnou matici, tak často vycházejí zlomky a tím pádem se dostáváme mimo obor $\mathbb{Z}$). Těleso netvoří ani čísla reprezentovaná na počítači v aritmetice s pohyblivou desetinnou čárkou — jednak nejsou operace sčítání a násobení uzavřené (pokud by výsledkem bylo hodně velké či hodně malé číslo), a jednak nejsou ani asociativní (díky zaokrouhlovávání).

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 4.26 — Kvaterniony)</span></p>

Dalším příkladem těles jsou *kvaterniony*. Jedná se o zobecnění komplexních čísel přidáním dalších dvou imaginárních jednotek $j$ a $k$, jejichž druhá mocnina je $-1$ a které jsou navíc svázány vztahem $ijk = -1$. Zatímco sčítání se definuje přirozeně, násobení je trochu komplikovanější a neplatí už pro něj komutativita. Kvaterniony pak tudíž tvoří nekomutativní těleso. Pomocí kvaternionů se dobře popisují rotace ve třírozměrném prostoru a našly využití i v robotice nebo ve fyzikální kvantové teorii.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Tvrzení 4.27 — Základní vlastnosti v tělese)</span></p>

Pro prvky tělesa platí následující vlastnosti:

1. $0a = 0$,
2. $ab = 0$ implikuje, že $a = 0$ nebo $b = 0$,
3. $-a = (-1)a$.

*Důkaz.* (1) Odvodíme $0a = (0 + 0)a = 0a + 0a$, přičtením $(-0a)$ dostaneme $0 = 0a$. (2) Je-li $a = 0$, pak věta platí. Je-li $a \neq 0$, pak existuje $a^{-1}$. Pronásobením obou stran rovnice $ab = 0$ zleva prvkem $a^{-1}$ dostaneme $a^{-1}ab = a^{-1}0$, neboli $1b = 0$, tj. $b = 0$. (3) Máme $0 = 0a = (1 - 1)a = 1a + (-1)a = a + (-1)a$, tedy $-a = (-1)a$.

</div>

Druhá vlastnost (a její důkaz) mj. říkají, že při rozhodování, zda nějaká struktura tvoří těleso, nemusíme ověřovat uzavřenost násobení na množině $\mathbb{T} \setminus \lbrace 0 \rbrace$ — tato vlastnost vyplývá z ostatních.

#### Konečná tělesa

Nyní se podíváme na konečná tělesa. Již v příkladu 4.3 jsme zavedli množinu $\mathbb{Z}_n = \lbrace 0, 1, \ldots, n - 1 \rbrace$. Operace $+$ a $\cdot$ na této množině definujeme modulo $n$. Snadno nahlédneme, že $\mathbb{Z}_2$ a $\mathbb{Z}_3$ jsou tělesy, ale $\mathbb{Z}_4$ už není, neboť prvek $2$ nemá inverzi $2^{-1}$. Tento výsledek můžeme zobecnit.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(Lemma 4.28)</span></p>

Buď $n$ prvočíslo a buď $0 \neq a \in \mathbb{Z}_n$. Pak při použití násobení modulo $n$ platí

$$\lbrace 0, 1, \ldots, n - 1 \rbrace = \lbrace 0a, 1a, \ldots, (n - 1)a \rbrace.$$

*Poznámka.* V množině $\lbrace 0a, 1a, \ldots, (n - 1)a \rbrace$ se tedy postupně objeví všechna čísla $0, 1, \ldots, n - 1$ (ne nutně v tomto pořadí) a každé z nich právě jednou.

*Důkaz.* Sporem předpokládejme, že $ak = a\ell$ pro nějaké $k, \ell \in \mathbb{Z}_n$, $k \neq \ell$. Pak dostáváme $a(k - \ell) = 0$, tudíž buď $a$ nebo $k - \ell$ je dělitelné $n$. To znamená buď $a = 0$ nebo $k - \ell = 0$. Ani jedna možnost ale nastat nemůže, což je spor.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Věta 4.29)</span></p>

$\mathbb{Z}_n$ je těleso právě tehdy, když $n$ je prvočíslo.

*Důkaz.* Je-li $n$ složené, pak $n = pq$, kde $1 < p, q < n$. Kdyby $\mathbb{Z}_n$ bylo těleso, pak $pq = 0$ implikuje podle tvrzení 4.27 buď $p = 0$ nebo $q = 0$, ale ani jedno neplatí.

Je-li $n$ prvočíslo, pak se snadno ověří všechny axiomy z definice tělesa. Jediný pracnější může být existence inverze $a^{-1}$ pro libovolné $a \neq 0$. To ale nahlédneme snadno z lemmatu 4.28. Protože $\lbrace 0, 1, \ldots, n - 1 \rbrace = \lbrace 0a, 1a, \ldots, (n - 1)a \rbrace$, musí být v množině napravo prvek $1$, a tudíž existuje $b \in \mathbb{Z}_n \setminus \lbrace 0 \rbrace$ takové, že $ba = 1$. Proto $b = a^{-1}$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 4.30 — Těleso $\mathbb{Z}_5$)</span></p>

Pro ilustraci uvádíme v tabulkách explicitní vyjádření obou operací nad tělesem $\mathbb{Z}_5$:

| $+$ | 0 | 1 | 2 | 3 | 4 |
|-----|---|---|---|---|---|
| **0** | 0 | 1 | 2 | 3 | 4 |
| **1** | 1 | 2 | 3 | 4 | 0 |
| **2** | 2 | 3 | 4 | 0 | 1 |
| **3** | 3 | 4 | 0 | 1 | 2 |
| **4** | 4 | 0 | 1 | 2 | 3 |

| $\cdot$ | 0 | 1 | 2 | 3 | 4 |
|---------|---|---|---|---|---|
| **0** | 0 | 0 | 0 | 0 | 0 |
| **1** | 0 | 1 | 2 | 3 | 4 |
| **2** | 0 | 2 | 4 | 1 | 3 |
| **3** | 0 | 3 | 1 | 4 | 2 |
| **4** | 0 | 4 | 3 | 2 | 1 |

Komutativita se projevuje jako symetrie tabulek, neutrální prvek kopíruje záhlaví tabulky do příslušného řádku a sloupce, a násobení nulou dává nulu. Vlastnost inverzního prvku se pak projevuje tak, že v každém řádku a sloupci (kromě násobení nulou) je uveden každý prvek tělesa právě jednou.

Inverzní prvky tělesa $\mathbb{Z}_5$ jsou pak:

| $x$ | 0 | 1 | 2 | 3 | 4 |
|-----|---|---|---|---|---|
| $-x$ | 0 | 4 | 3 | 2 | 1 |

| $x$ | 1 | 2 | 3 | 4 |
|-----|---|---|---|---|
| $x^{-1}$ | 1 | 3 | 2 | 4 |

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 4.31 — Těleso $\mathbb{Z}_2$ a bity)</span></p>

Těleso $\mathbb{Z}_2$ má pro informatiky obzvláště velký význam, protože pracuje se dvěma prvky $0$ a $1$, na které můžeme nahlížet jako na počítačové bity. Mnoho běžných operací s bity pak lze přeložit v řeči operací v tělese $\mathbb{Z}_2$. Je snadné nahlédnout, že operace sčítání v $\mathbb{Z}_2$ odpovídá počítačové operaci XOR a násobení odpovídá operaci AND. Podobně i ostatní logické operace můžeme vyjádřit pomocí operací v tělese $\mathbb{Z}_2$. Tím pádem jakýkoliv logický člen digitálního obvodu reprezentuje nějaký aritmetický výraz nad $\mathbb{Z}_2$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Poznámka 4.32 — Matice nad tělesem)</span></p>

Soustavy rovnic a operace s maticemi jsme zaváděli nad tělesem reálných čísel. Nicméně nic nám nebrání rozšířit tyto pojmy a pracovat nad jakýmkoli jiným tělesem. Je-li $\mathbb{T}$ těleso, pak $\mathbb{T}^{m \times n}$ bude značit matici řádu $m \times n$ s prvky v tělese $\mathbb{T}$. Jediné vlastnosti reálných čísel, který jsme používali, jsou přesně ty, které se vyskytují v definici tělesa. Proto veškeré postupy a teorie vybudované v předchozích kapitolách 2 a 3 zůstanou v platnosti. Můžeme tak například řešit soustavy lineárních rovnic nad libovolným tělesem pomocí Gaussovy eliminace, hovořit o regularitě matice z $\mathbb{T}^{n \times n}$ či hledat její inverzi.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 4.33 — Výpočet inverzní matice nad $\mathbb{Z}_5$)</span></p>

$$(A \mid I_3) = \begin{pmatrix} 1 & 2 & 3 & 1 & 0 & 0 \\ 2 & 0 & 4 & 0 & 1 & 0 \\ 3 & 3 & 4 & 0 & 0 & 1 \end{pmatrix} \sim \ldots \sim \begin{pmatrix} 1 & 0 & 0 & 2 & 4 & 2 \\ 0 & 1 & 0 & 1 & 0 & 3 \\ 0 & 0 & 1 & 4 & 2 & 4 \end{pmatrix} = (I_3 \mid A^{-1})$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Poznámka 4.34 — Jak najít inverzi)</span></p>

Přirozená otázka při počítání nad tělesem $\mathbb{Z}_p$ zní, jak najít inverzní prvek k prvku $x \in \mathbb{Z}_p \setminus \lbrace 0 \rbrace$. Pro malé hodnoty $p$ mohu zkusit postupně $1, 2, \ldots, p - 1$ dokud nenarazím na inverzní prvek k $x$. Pokud $p$ je hodně velké prvočíslo, tento postup už není efektivní a postupuje se tzv. *rozšířeným Eukleidovým algoritmem*, který najde $a, b \in \mathbb{Z}$ taková, že $ax + bp = 1$. Z rovnice vidíme, že hledanou inverzí $x^{-1}$ je prvek $a$, bereme-li jeho zbytek po dělení $p$.

</div>

#### Velikosti konečných těles

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Věta 4.35 — O velikostech konečných těles)</span></p>

Existují konečná tělesa právě o velikostech $p^n$, kde $p$ je prvočíslo a $n \ge 1$.

</div>

Důkaz vynecháme, ale ukážeme základní myšlenku, jak sestrojit těleso o velikosti $p^n$. Takové těleso se značí symbolem $\operatorname{GF}(p^n)$ (Galois field) a jeho prvky jsou polynomy stupně nanejvýš $n - 1$ s koeficienty v tělese $\mathbb{Z}_p$, tedy

$$\operatorname{GF}(p^n) = \lbrace a_{n-1}x^{n-1} + \ldots + a_1 x + a_0 ;\ a_0, \ldots, a_{n-1} \in \mathbb{Z}_p \rbrace.$$

Sčítání je definováno analogicky jako pro reálné polynomy. Násobením bychom však mohli dostat polynomy vyšších stupňů než $n - 1$. Proto nejprve zvolíme libovolný pevný ireducibilní polynom stupně $n$ a polynomy vynásobíme běžným způsobem a pak vezmeme zbytek při dělení tímto ireducibilním polynomem.

Další zajímavá vlastnost je, že každé konečné těleso velikosti $p^n$ je isomorfní s $\operatorname{GF}(p^n)$, to znamená, že taková tělesa jsou v zásadě stejná až na jiné označení prvků.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 4.36 — Těleso $\operatorname{GF}(8)$)</span></p>

Množina má za prvky polynomy stupňů nanejvýš dva s koeficienty v $\mathbb{Z}_2$:

$$\operatorname{GF}(8) = \lbrace 0,\ 1,\ x,\ x + 1,\ x^2,\ x^2 + 1,\ x^2 + x,\ x^2 + x + 1 \rbrace.$$

Sčítání je definované $(a_2 x^2 + a_1 x + a_0) + (b_2 x^2 + b_1 x + b_0) = (a_2 + b_2)x^2 + (a_1 + b_1)x + (a_0 + b_0)$, např. $(x + 1) + (x^2 + x) = x^2 + 1$. Uvažme ireducibilní polynom $x^3 + x + 1$. Pak násobíme modulo tento polynom, např. $x^2 \cdot x = -x - 1 = x + 1$, nebo $x^2 \cdot (x^2 + 1) = -x = x$.

</div>

#### Charakteristika tělesa

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Definice 4.37 — Charakteristika tělesa)</span></p>

*Charakteristika tělesa* $\mathbb{T}$ je nejmenší $n$ takové, že

$$\underbrace{1 + 1 + \ldots + 1}_{n} = 0.$$

Pokud takové $n$ neexistuje, pak ji definujeme jako $0$.

</div>

Kupříkladu nekonečná tělesa $\mathbb{Q}$, $\mathbb{R}$ či $\mathbb{C}$ mají charakteristiku $0$, těleso $\mathbb{Z}_p$ má charakteristiku $p$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Tvrzení 4.38)</span></p>

Charakteristika tělesa je buď nula, nebo prvočíslo.

*Důkaz.* Protože $0 \neq 1$, charakteristika nemůže být $1$. Pokud by byla charakteristika složené číslo $n = pq$, pak

$$0 = \underbrace{1 + 1 + \ldots + 1}\_{n = pq} = (\underbrace{1 + \ldots + 1}\_{p})(\underbrace{1 + \ldots + 1}\_{q}),$$

tedy součet $p$ nebo $q$ jedniček dá nulu, což je spor s minimalitou $n$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Poznámka 4.39 — Průměr v tělese)</span></p>

Jestliže charakteristika tělesa $\mathbb{T}$ není $2$, tak můžeme zavést něco jako průměr. Označme symbolem $2$ hodnotu $1 + 1$ a pak pro libovolné $a, b \in \mathbb{T}$ má číslo $p = \tfrac{1}{2}(a + b)$ vlastnost $a - p = p - b$, je tedy stejně „vzdálené" od $a$ jako od $b$.

Těleso s charakteristikou $2$ je $\mathbb{Z}_2$ nebo obecněji jakékoliv těleso $\operatorname{GF}(2^n)$. V těchto tělesech tedy průměr $0$ a $1$ nelze zadefinovat, zatímco například v tělese $\mathbb{Z}_5$ je průměr čísel $0$ a $1$ číslo $3$.

</div>

#### Malá Fermatova věta

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Věta 4.40 — Malá Fermatova věta)</span></p>

Buď $p$ prvočíslo a buď $0 \neq a \in \mathbb{Z}_p$. Pak $a^{p-1} = 1$ v tělese $\mathbb{Z}_p$.

*Důkaz.* Podle lemmatu 4.28 je $\lbrace 0, 1, \ldots, p - 1 \rbrace = \lbrace 0a, 1a, \ldots, (p - 1)a \rbrace$. Protože $0 = 0a$, tak dostáváme $\lbrace 1, \ldots, p - 1 \rbrace = \lbrace 1a, \ldots, (p - 1)a \rbrace$. Tudíž $1 \cdot 2 \cdot 3 \cdot \ldots \cdot (p - 1) = (1a) \cdot (2a) \cdot (3a) \cdot \ldots \cdot (p - 1)a$. Zkrácením obou stran čísly $1, 2, \ldots, p - 1$ získáme požadovanou rovnost $1 = a \cdot \ldots \cdot a = a^{p-1}$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 4.41)</span></p>

Jaká je hodnota $2^{111}$ v tělese $\mathbb{Z}_{11}$? Podle Malé Fermatovy věty je $2^{10} = 1$, tudíž i $2^{110} = 1$. Proto $2^{111} = 2^{110+1} = 2^{110} \cdot 2^1 = 2$.

</div>

### 4.4 Aplikace

Konečná tělesa se používají například v kódování a šifrování. Na závěr této kapitoly ukážeme praktické využití těles právě v kódování.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 4.42 — Samoopravné kódy — Hammingův kód $(7, 4, 3)$)</span></p>

Uvažujme problém přenosu dat, která jsou tvořena posloupností nul a jedniček. Zatímco úlohou šifrování je transformovat data tak, aby je nikdo nepovolaný nepřečetl, úlohou kódování je zlepšit jejich přenosové vlastnosti. Tím myslíme zejména umět detekovat a opravit chyby, které při přenosu přirozeně vznikají.

Hammingův kód $(7, 4, 3)$ spočívá v rozdělení přenosových dat na úseky o čtyřech bitech, které zakódujeme na sedm bitů. Tento kód umí detekovat a opravit jednu přenosovou chybu. Kódování a dekódování jde elegantně reprezentovat maticovým násobením. Úsek čtyř bitů si představíme jako aritmetický vektor $a$ nad tělesem $\mathbb{Z}_2$. Kódování probíhá vynásobením vektoru $a$ takzvanou generující maticí $H \in \mathbb{Z}_2^{7 \times 4}$:

$$Ha = b.$$

Příjemce obdrží úsek reprezentovaný vektorem $b$. Bity původních dat jsou na zvýrazněných pozicích $b_3, b_5, b_6, b_7$, ostatní bity $b_1, b_2, b_4$ jsou kontrolní. K detekci a opravě chyb používá příjemce detekční matici $D \in \mathbb{Z}_2^{3 \times 7}$. Pokud $Db = 0$, nedošlo k žádné chybě v přenosu (nebo nastaly více než dvě chyby). V opačném případě nastala přenosová chyba a chybný bit je na pozici $Db$, bereme-li tento vektor jako binární zápis přirozeného čísla.

</div>

### Shrnutí ke kapitole 4

Grupy představují první axiomaticky definovaný abstraktní pojem, se kterým jsme se setkali. Grupa je jakákoli množina, na které máme zavedenou operaci splňující několik základních vlastností (asociativita, neutrální a inverzní prvek, případně komutativita). Právě tato abstraktní definice umožňuje obsáhnout velkou řadu objektů a tak rozšiřuje pole působnosti. Jako význačný příklad nekomutativní grupy jsme probírali permutace s operací skládání, tzv. symetrickou grupu, protože právě k popisu symetrií byla vymyšlena.

Algebraická tělesa jsou oproti grupám bohatší o další operaci. Tělesem je tedy množina se dvěma operacemi, splňujícími určité vlastnosti. Maticové operace probírané v minulých kapitolách tak lze směle rozšířit a pracovat nad libovolným tělesem, nikoliv jen nad $\mathbb{R}$; veškeré výsledky zůstanou v zásadě v platnosti. Známe nekonečná tělesa jako například $\mathbb{R}$ či $\mathbb{C}$, a konečná tělesa jako například informatikům blízké dvouprvkové těleso $\mathbb{Z}_2$. Konečná tělesa existují právě o velikosti $p^n$, kde $p$ je prvočíslo.

## Kapitola 5 — Vektorové prostory

Vektorové prostory (v některých odvětvích označované také jako *lineární prostory*) zobecňují dobře známý prostor aritmetických vektorů $\mathbb{R}^n$. Stejně jako u grup a těles je zadefinujeme pomocí abstraktních axiomů.

### 5.1 Základní pojmy

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 5.1 — Motivační)</span></p>

Uspořádaná $n$-tice reálných čísel $v = (v_1, \ldots, v_n)$ má v eukleidovském $n$-dimenzionálním prostoru $\mathbb{R}^n$ dvě možné interpretace a obě budeme pro geometrickou představu používat. Můžeme se na ni dívat jako na jeden konkrétní bod nebo jako na vektor. Vektor udává směr od počátku $(0, \ldots, 0)$ k bodu $(v_1, \ldots, v_n)$. S vektory umíme následující operace:

- *Sčítání.* Součtem vektorů je opět vektor, pro $u, v \in \mathbb{R}^n$ je $u + v = (u_1 + v_1, \ldots, u_n + v_n)$. Sčítání je komutativní a asociativní.
- *Násobení číslem.* Násobek vektoru je opět vektor, pro $\alpha \in \mathbb{R}$, $v \in \mathbb{R}^n$ je $\alpha v = (\alpha v_1, \ldots, \alpha v_n)$. Násobek vektoru udává stejný směr (pokud $\alpha > 0$) nebo opačný směr (pokud $\alpha < 0$). Jsou splněny základní vlastnosti jako například distributivita vůči sčítání.

</div>

S reálnými aritmetickými vektory jsou možné ještě další operace, ale ty prozatím neuvažujeme. V naší snaze zobecnit pojem vektoru a prostoru vektorů budeme přirozeně požadovat podobné vlastnosti, které jsme zmínili nahoře. Tedy abychom vektory uměli sčítat a násobit skalárem (číslem) a aby tyto operace splňovaly základní axiomy.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Definice 5.2 — Vektorový prostor)</span></p>

Buď $\mathbb{T}$ těleso s neutrálními prvky $0$ pro sčítání a $1$ pro násobení. *Vektorovým prostorem nad tělesem* $\mathbb{T}$ rozumíme množinu $V$ s operacemi sčítání vektorů $+ \colon V^2 \to V$, a násobení vektoru skalárem $\cdot \colon \mathbb{T} \times V \to V$ splňující pro každé $\alpha, \beta \in \mathbb{T}$ a $u, v \in V$:

1. $(V, +)$ je Abelova grupa, neutrální prvek značíme $o$ a inverzní k $v$ pak $-v$,
2. $\alpha(\beta v) = (\alpha \beta)v$ &emsp; (asociativita),
3. $1v = v$,
4. $(\alpha + \beta)v = \alpha v + \beta v$ &emsp; (distributivita),
5. $\alpha(u + v) = \alpha u + \alpha v$ &emsp; (distributivita).

Prvkům vektorového prostoru $V$ říkáme *vektory* a budeme je značit latinkou. Vektory píšeme bez šipek, tedy $v$ a ne $\vec{v}$. Prvkům tělesa $\mathbb{T}$ pak říkáme *skaláry*, a pro odlišení je budeme značit řeckými písmeny.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 5.3 — Příklady vektorových prostorů)</span></p>

- Aritmetický prostor $\mathbb{R}^n$ nad $\mathbb{R}$, či obecněji $\mathbb{T}^n$ nad $\mathbb{T}$, kde $\mathbb{T}$ je libovolné těleso; $n$-tice prvků z tělesa $\mathbb{T}$ sčítáme a násobíme skalárem po složkách podobně jako u $\mathbb{R}^n$. Axiomy z definice vektorového prostoru pak vyplývají z vlastností tělesa.
- Prostor matic $\mathbb{R}^{m \times n}$ nad $\mathbb{R}$, či obecněji $\mathbb{T}^{m \times n}$ nad $\mathbb{T}$. Axiomy z definice vektorového prostoru se snadno nahlédnou z vlastností matic a těles.
- Prostor všech reálných polynomů proměnné $x$ nad tělesem $\mathbb{R}$, který značíme $\mathcal{P}$.
- Prostor všech reálných polynomů nad $\mathbb{R}$ proměnné $x$ stupně nanejvýš $n$, který značíme $\mathcal{P}^n$. Operace jsou definovány standardním způsobem:
  - Sčítání: $(a_n x^n + \ldots + a_1 x + a_0) + (b_n x^n + \ldots + b_1 x + b_0) = (a_n + b_n)x^n + \ldots + (a_1 + b_1)x + (a_0 + b_0)$
  - Násobení skalárem: $\alpha(a_n x^n + \ldots + a_1 x + a_0) = (\alpha a_n)x^n + \ldots + (\alpha a_1)x + (\alpha a_0)$
  - Nulový vektor: $0$. Opačný vektor: $-(a_n x^n + \ldots + a_0) = (-a_n)x^n + \ldots + (-a_0)$.
- Prostor všech reálných funkcí $f \colon \mathbb{R} \to \mathbb{R}$, který značíme $\mathcal{F}$. Funkce $f, g \colon \mathbb{R} \to \mathbb{R}$ sčítáme tak, že sečteme příslušné funkční hodnoty, tedy $(f + g)(x) = f(x) + g(x)$. Podobně funkci $f \colon \mathbb{R} \to \mathbb{R}$ násobíme skalárem $\alpha \in \mathbb{R}$ tak, že vynásobíme všechny funkční hodnoty, tj. $(\alpha f)(x) = \alpha f(x)$.
- Prostor všech spojitých funkcí $f \colon \mathbb{R} \to \mathbb{R}$, který značíme $\mathcal{C}$. Prostor všech spojitých funkcí $f \colon [a, b] \to \mathbb{R}$ na intervalu $[a, b]$ pak značíme $\mathcal{C}_{[a,b]}$. Operace jsou definovány analogicky jako pro $\mathcal{F}$.

Pokud neřekneme jinak, prostory $\mathbb{R}^n$ a $\mathbb{R}^{m \times n}$ budeme nadále implicitně uvažovat nad tělesem $\mathbb{R}$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Tvrzení 5.4 — Základní vlastnosti vektorů)</span></p>

V prostoru $V$ nad tělesem $\mathbb{T}$ platí pro každý skalár $\alpha \in \mathbb{T}$ a vektor $v \in V$:

1. $0v = o$,
2. $\alpha o = o$,
3. $\alpha v = o$ implikuje, že $\alpha = 0$ nebo $v = o$,
4. $(-1)v = -v$.

*Důkaz.* Analogicky jako u vlastností v tělese.

</div>

### 5.2 Podprostory a lineární kombinace

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Definice 5.5 — Podprostor)</span></p>

Buď $V$ vektorový prostor nad $\mathbb{T}$. Pak $U \subseteq V$ je *podprostorem* prostoru $V$, pokud tvoří vektorový prostor nad $\mathbb{T}$ se stejně definovanými operacemi. Značení: $U \le V$.

</div>

Jak ukazuje následující tvrzení, ekvivalentní definice podprostoru je, že musí obsahovat nulový vektor a být uzavřený na obě operace.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Tvrzení 5.6)</span></p>

Buď $U$ podmnožina vektorového prostoru $V$ nad $\mathbb{T}$. Pak $U$ je podprostorem $V$ právě tehdy, když platí:

1. $o \in U$,
2. $\forall u, v \in U: u + v \in U$,
3. $\forall \alpha \in \mathbb{T}\ \forall u \in U: \alpha u \in U$.

*Důkaz.* Pokud je $U$ podprostorem $V$, pak musí splňovat požadované tři vlastnosti z definice vektorového prostoru. Naopak, předpokládejme, že $U$ splňuje zadané tři vlastnosti. Ostatní vlastnosti z definice vektorového prostoru (jako je komutativita, asociativita, distributivita) pak platí také, protože platí pro množinu $V$, a tudíž automaticky platí i pro každou její podmnožinu. To, že je množina $U$ uzavřená na opačné vektory, vyplývá z uzavřenosti na násobky, neboť podle tvrzení 5.4 je $-v = (-1)v$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 5.7 — Příklady vektorových podprostorů)</span></p>

- Dva triviální podprostory prostoru $V$ jsou: $V$ a $\lbrace o \rbrace$.
- Libovolná přímka v rovině procházející počátkem je podprostorem $\mathbb{R}^2$, jiná ne.
- $\mathcal{P}^n \le \mathcal{P} \le \mathcal{C} \le \mathcal{F}$.
- Množina symetrických reálných matic řádu $n$ je podprostorem prostoru $\mathbb{R}^{n \times n}$.
- $\mathbb{Q}^n$ nad $\mathbb{Q}$ je podprostorem prostoru $\mathbb{R}^n$ nad $\mathbb{Q}$, ale není podprostorem prostoru $\mathbb{R}^n$ nad $\mathbb{R}$, protože pracuje nad jiným tělesem.

Některé vlastnosti vektorových podprostorů:

- Jsou-li $U, V$ podprostory prostoru $W$ a platí-li $U \subseteq V$, pak $U \le V$.
- Pro vlastnost „býti podprostorem" platí transitivita, čili $U \le V \le W$ implikuje $U \le W$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Tvrzení 5.8 — Průnik podprostorů)</span></p>

Buď $V$ vektorový prostor nad $\mathbb{T}$, a mějme $V_i$, $i \in I$, libovolný systém podprostorů $V$. Pak $\bigcap_{i \in I} V_i$ je opět podprostor $V$.

*Důkaz.* Podle tvrzení 5.6 stačí ověřit tři vlastnosti: Protože $o \in V_i$ pro každé $i \in I$, musí být i v jejich průniku. Uzavřenost na sčítání: Buď $u, v \in \bigcap_{i \in I} V_i$, tj. pro každé $i \in I$ je $u, v \in V_i$, tedy i $u + v \in V_i$. Proto $u + v \in \bigcap_{i \in I} V_i$. Analogicky uzavřenost na násobky.

</div>

#### Lineární obal

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Definice 5.9 — Lineární obal)</span></p>

Buď $V$ vektorový prostor nad $\mathbb{T}$, a $W \subseteq V$. Pak *lineární obal* $W$, značený $\operatorname{span}(W)$, je průnik všech podprostorů $V$ obsahujících $W$, to jest $\operatorname{span}(W) = \bigcap_{U: W \subseteq U \le V} U$.

</div>

Lineární obal množiny vektorů $W$ je tedy nejmenší prostor obsahující $W$ v tom smyslu, že jakýkoli jiný prostor obsahující $W$ je jeho nadmnožinou.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 5.10 — Příklady lineárních obalů)</span></p>

Příklady lineárních obalů ve vektorovém prostoru $\mathbb{R}^2$:

- $\operatorname{span}\lbrace (1, 0)^T \rbrace$ je přímka, konkrétně osa $x_1$.
- $\operatorname{span}\lbrace (1, 0)^T, (2, 0)^T \rbrace$ je totéž.
- $\operatorname{span}\lbrace (1, 1)^T, (1, 2)^T \rbrace$ je celá rovina $\mathbb{R}^2$.
- $\operatorname{span}\lbrace \rbrace = \lbrace o \rbrace$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Definice 5.11 — Generátory a konečně generovaný prostor)</span></p>

Nechť prostor $U$ je lineárním obalem množiny vektorů $W$, tedy $U = \operatorname{span}(W)$. Pak říkáme, že $W$ *generuje* prostor $U$, a prvky množiny $W$ jsou *generátory* prostoru $U$. Prostor $U$ se nazývá *konečně generovaný*, jestliže je generovaný nějakou konečnou množinou vektorů.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 5.12)</span></p>

Uvažujme vektorový prostor $\mathbb{R}^2$ a jeho podprostor $U$ reprezentovaný osou $x_1$. Tento podprostor lze vygenerovat vektorem $(1, 0)^T$, nebo lze vygenerovat vektorem $(-3, 0)^T$, anebo jakýmkoli jiným vektorem tvaru $(a, 0)^T$, kde $a \neq 0$. Nicméně, $U$ lze vygenerovat i množinou vektorů $\lbrace (2, 0)^T, (5, 0)^T \rbrace$. Vidíme, že tato množina není minimální — jeden vektor lze odstranit a zbylý vektor stále generuje podprostor $U$. Tato snaha o minimální reprezentaci a odstranění redundancí povede později k pojmu báze (sekce 5.4).

</div>

#### Lineární kombinace

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Definice 5.13 — Lineární kombinace)</span></p>

Buď $V$ vektorový prostor nad $\mathbb{T}$ a $v_1, \ldots, v_n \in V$. Pak *lineární kombinací* vektorů $v_1, \ldots, v_n$ rozumíme výraz typu $\sum_{i=1}^{n} \alpha_i v_i = \alpha_1 v_1 + \ldots + \alpha_n v_n$, kde $\alpha_1, \ldots, \alpha_n \in \mathbb{T}$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Poznámka 5.14)</span></p>

Zde je potřeba zdůraznit, že uvažujeme pouze lineární kombinace konečně mnoha vektorů. To pro naše účely plně postačuje, protože vesměs budeme pracovat s konečně generovanými vektorovými prostory. Nekonečné lineární kombinace je možné také v některých případech zavést, ale potřebovali bychom silnější předpoklady (např. pracovat nad $\mathbb{R}$) a silnější aparát (limity, konvergenci, ...).

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Poznámka 5.15)</span></p>

Lineární kombinaci lze chápat dvěma způsoby. První způsob je chápat ji jako výraz $\sum_{i=1}^{n} \alpha_i v_i$ a druhý způsob je uvažovat její konkrétní hodnotu, tedy výsledný vektor. Budeme používat oba tyto pohledy.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Věta 5.17)</span></p>

Buď $V$ vektorový prostor nad $\mathbb{T}$, a mějme $v_1, \ldots, v_n \in V$. Pak

$$\operatorname{span}\lbrace v_1, \ldots, v_n \rbrace = \lbrace \sum_{i=1}^{n} \alpha_i v_i ;\ \alpha_1, \ldots, \alpha_n \in \mathbb{T} \rbrace.$$

*Důkaz.* Inkluze „$\supseteq$". Lineární obal $\operatorname{span}\lbrace v_1, \ldots, v_n \rbrace$ je podprostor $V$ obsahující vektory $v_1, \ldots, v_n$, tedy musí být uzavřený na násobky a součty. Tudíž obsahuje i násobky $\alpha_i v_i$, $i = 1, \ldots, n$, a také jejich součet $\sum_{i=1}^{n} \alpha_i v_i$.

Inkluze „$\subseteq$". Stačí ukázat, že množina lineárních kombinací $M := \lbrace \sum_{i=1}^{n} \alpha_i v_i ;\ \alpha_1, \ldots, \alpha_n \in \mathbb{T} \rbrace$ je vektorový podprostor $V$ obsahující vektory $v_1, \ldots, v_n$, a proto je jednou z množin, jejichž průniku je $\operatorname{span}\lbrace v_1, \ldots, v_n \rbrace$. Pro každé $v_i$ v množině $M$ obsažen, stačí vzít lineární kombinaci s $\alpha_i = 1$ a $\alpha_j = 0$, $j \neq i$. Nulový vektor rovněž obsahuje, vezměme lineární kombinaci s nulovými koeficienty. Uzavřenost na součty: Vezměme libovolné dva vektory $u = \sum_{i=1}^{n} \beta_i v_i$, $u' = \sum_{i=1}^{n} \beta_i' v_i$ z množiny $M$. Pak $u + u' = \sum_{i=1}^{n} (\beta_i + \beta_i') v_i$, což je prvek množiny $M$. Podobně pro násobky.

</div>

Lineární obal jednoho vektoru $v$ je dán množinou všech jeho lineárních kombinací, tedy jeho násobků. Lineární obal dvou vektorů $u, v$ (s různými směry) v prostoru $\mathbb{R}^3$ představuje rovinu.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Tvrzení 5.19)</span></p>

Buď $V$ vektorový prostor nad $\mathbb{T}$ a buď $M \subseteq V$. Pak $\operatorname{span}(M)$ je tvořen všemi lineárními kombinacemi každé konečné soustavy vektorů z $M$.

*Důkaz.* Analogický důkazu věty 5.17, necháváme na cvičení.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Poznámka 5.20 — Trochu jiný pohled na soustavu rovnic)</span></p>

Výraz $Ax = \sum_j x_j A_{*j}$ je vlastně lineární kombinace sloupců matice $A$ (srov. poznámka 3.19), takže řešit soustavu $Ax = b$ znamená hledat lineární kombinaci sloupců, která se rovná $b$. Řešení tedy existuje právě tehdy, když $b$ náleží do podprostoru generovaného sloupci matice $A$, tedy $b \in \operatorname{span}\lbrace A_{*1}, \ldots, A_{*n} \rbrace$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Poznámka 5.21 — Trochu jiný pohled na součin matic)</span></p>

Předchozí úvahu můžeme použít i pro maticové násobení. Uvažujme $A \in \mathbb{T}^{m \times p}$, $B \in \mathbb{T}^{p \times n}$. Zaměříme se nejprve na sloupce výsledné matice $AB$. Libovolný $j$-tý sloupec vyjádříme $(AB)_{*j} = AB_{*j} = \sum_{k=1}^{p} b_{kj} A_{*k}$, je tedy lineární kombinací sloupců matice $A$. Každý sloupec matice $AB$ je tudíž tvořen lineární kombinací sloupců matice $A$.

Podobně lze interpretovat maticové násobení jako vytváření lineárních kombinací řádků. Libovolný $i$-tý řádek výsledné matice $AB$ vyjádříme jako $(AB)_{i*} = A_{i*}B = \sum_{k=1}^{p} a_{ik}B_{k*}$, a tedy představuje lineární kombinaci řádků matice $B$. Na elementární řádkové úpravy matice $B$ se pak můžeme dívat jako na vytváření lineárních kombinací řádků a nahrazování původních řádků těmito kombinacemi.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Poznámka 5.22 — Ještě jiný pohled na součin matic)</span></p>

Součin $A \in \mathbb{T}^{m \times p}$, $B \in \mathbb{T}^{p \times n}$ lze vyjádřit ještě jiným způsobem jako $AB = \sum_{k=1}^{p} A_{*k}B_{k*}$. Každý člen sumy představuje vnější součin dvou vektorů, což vytvoří matici hodnosti nanejvýš $1$. Tímto předpisem jsme tedy rozepsali matici na součet maximálně $k$ matic hodnosti $1$.

</div>

### 5.3 Lineární nezávislost

Konečně generovaný prostor typicky může být generován různými množinami vektorů. Motivací pro tuto sekci je snaha najít množinu generátorů, která bude minimální co do počtu i co do inkluze (tedy žádná ostrá podmnožina už prostor negeneruje), srov. příklad 5.12. To pak povede i k pojmům jako báze, souřadnice a dimenze.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Definice 5.23 — Lineární nezávislost)</span></p>

Buď $V$ vektorový prostor nad $\mathbb{T}$ a mějme vektory $v_1, \ldots, v_n \in V$. Pak vektory $v_1, \ldots, v_n$ se nazývají *lineárně nezávislé*, pokud rovnost $\sum_{i=1}^{n} \alpha_i v_i = o$ nastane pouze pro $\alpha_1 = \ldots = \alpha_n = 0$. V opačném případě jsou vektory *lineárně závislé*.

</div>

Tedy vektory jsou lineárně závislé, pokud existují $\alpha_1, \ldots, \alpha_n \in \mathbb{T}$, ne všechna nulová a taková, že $\sum_{i=1}^{n} \alpha_i v_i = o$.

Pojem lineární nezávislosti zobecníme i na nekonečné množiny vektorů — nicméně s nekonečny bývá trochu potíž (např. co by se myslelo nekonečnou lineární kombinací?), proto se to definuje takto:

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Definice 5.24 — Lineární nezávislost nekonečné množiny)</span></p>

Buď $V$ vektorový prostor nad $\mathbb{T}$ a buď $M \subseteq V$ nekonečná množina vektorů. Pak $M$ je *lineárně nezávislá*, pokud každá konečná podmnožina $M$ je lineárně nezávislá. V opačném případě je $M$ *lineárně závislá*.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 5.25 — Příklady lineárně (ne)závislých vektorů v $\mathbb{R}^2$)</span></p>

- $(1, 0)^T$ je lineárně nezávislý,
- $(1, 0)^T$, $(2, 0)^T$ jsou lineárně závislé,
- $(1, 1)^T$, $(1, 2)^T$ jsou lineárně nezávislé,
- $(1, 0)^T$, $(0, 1)^T$, $(1, 1)^T$ jsou lineárně závislé,
- $(0, 0)^T$ je lineárně závislý,
- prázdná množina je lineárně nezávislá.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 5.26 — Testování lineární nezávislosti)</span></p>

Není těžké nahlédnout, že dva vektory tvoří lineárně závislý systém pokud je jeden z nich násobkem druhého. Pro více vektorů však lineární závislost není tak snadno vidět. Jak prakticky zjistit, zda dané aritmetické vektory, např. $(1, 3, 2)^T$, $(2, 5, 3)^T$, $(2, 3, 1)^T$, jsou lineárně závislé či nezávislé? Podle definice hledejme, kdy lineární kombinace vektorů dá nulový vektor:

$$\alpha \begin{pmatrix} 1 \\ 3 \\ 2 \end{pmatrix} + \beta \begin{pmatrix} 2 \\ 5 \\ 3 \end{pmatrix} + \gamma \begin{pmatrix} 2 \\ 3 \\ 1 \end{pmatrix} = \begin{pmatrix} 0 \\ 0 \\ 0 \end{pmatrix}.$$

Toto vyjádříme ekvivalentně jako soustavu rovnic s neznámými $\alpha, \beta, \gamma$ a vyřešíme úpravou matice soustavy na odstupňovaný tvar:

$$\begin{pmatrix} 1 & 2 & 2 \\ 3 & 5 & 3 \\ 2 & 3 & 1 \end{pmatrix} \sim \begin{pmatrix} 1 & 0 & -4 \\ 0 & 1 & 3 \\ 0 & 0 & 0 \end{pmatrix}.$$

Soustava má nekonečně mnoho řešení a určitě najdeme nějaké nenulové, např. $\alpha = 4$, $\beta = -3$, $\gamma = 1$. To znamená, že dané vektory jsou lineárně závislé.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Příklad 5.27 — Souvislost s regularitou)</span></p>

Definice lineární nezávislosti trochu připomíná definici regularity (definice 3.26). Není to náhoda, sloupce regulární matice (a potažmo i řádky) představují další příklad lineárně nezávislých vektorů. Podle definice je čtvercová matice $A$ regulární, pokud rovnost $\sum_j A_{*j} x_j = o$ nastane pouze pro $x = o$, a toto přesně odpovídá lineární nezávislosti sloupců matice $A$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Věta 5.28 — Charakterizace lineární závislosti)</span></p>

Buď $V$ vektorový prostor nad $\mathbb{T}$, a mějme $v_1, \ldots, v_n \in V$. Pak vektory $v_1, \ldots, v_n$ jsou lineárně závislé právě tehdy, když existuje $k \in \lbrace 1, \ldots, n \rbrace$ takové, že $v_k = \sum_{i \neq k} \alpha_i v_i$ pro nějaké $\alpha_1, \ldots, \alpha_n \in \mathbb{T}$, to jest $v_k \in \operatorname{span}\lbrace v_1, \ldots, v_{k-1}, v_{k+1}, \ldots, v_n \rbrace$.

*Důkaz.* Implikace „$\Rightarrow$". Jsou-li vektory lineárně závislé, pak existuje jejich netriviální lineární kombinace rovna nule, tj. $\sum_{i=1}^{n} \beta_i v_i = o$ pro $\beta_1, \ldots, \beta_n \in \mathbb{T}$ a $\beta_k \neq 0$ pro nějaké $k \in \lbrace 1, \ldots, n \rbrace$. Zde můžeme zvolit libovolné $k$ takové, že $\beta_k \neq 0$. Vyjádříme $k$-tý člen $\beta_k v_k = -\sum_{i \neq k} \beta_i v_i$ a po zkrácení dostáváme požadovaný předpis $v_k = \sum_{i \neq k} (-\beta_k^{-1}\beta_i)v_i$.

Implikace „$\Leftarrow$". Je-li $v_k = \sum_{i \neq k} \alpha_i v_i$, pak $v_k - \sum_{i \neq k} \alpha_i v_i = o$, což je požadovaná netriviální kombinace rovna nule, neboť koeficient u $v_k$ je $1 \neq 0$.

</div>

Důsledkem je ještě jiná charakterizace lineární závislosti. Ta mj. říká, že vektory jsou lineárně závislé právě tehdy, když odebráním nějakého (ale ne libovolného, viz příklad 5.30) z nich se jejich lineární obal nezmenší. Tudíž mezi nimi je nějaký nadbytečný. U lineárně nezávislého systému je tomu naopak: odebráním libovolného z nich se jejich lineární obal ostře zmenší, není mezi nimi tedy žádný nadbytečný.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(Důsledek 5.29)</span></p>

Buď $V$ vektorový prostor nad $\mathbb{T}$, a mějme $v_1, \ldots, v_n \in V$. Pak vektory $v_1, \ldots, v_n$ jsou lineárně závislé právě tehdy, když existuje $k \in \lbrace 1, \ldots, n \rbrace$ takové, že

$$\operatorname{span}\lbrace v_1, \ldots, v_n \rbrace = \operatorname{span}\lbrace v_1, \ldots, v_{k-1}, v_{k+1}, \ldots, v_n \rbrace.$$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 5.30)</span></p>

Vektory $(2, 3)^T, (2, 1)^T, (4, 2)^T \in \mathbb{R}^2$ jsou lineárně závislé, tudíž jejich lineární obal lze vygenerovat i z vlastní podmnožiny těchto vektorů. Můžeme odstranit například druhý, anebo třetí vektor (ale ne oba zároveň) a výsledné dva vektory pořád budou generovat stejný prostor $\mathbb{R}^2$. Nicméně, první vektor odebrat nelze, zbylé dva vektory už $\mathbb{R}^2$ nevygenerují!

</div>

### 5.4 Báze

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Definice 5.31 — Báze)</span></p>

Buď $V$ vektorový prostor nad $\mathbb{T}$. Pak *bází* rozumíme jakýkoli lineárně nezávislý systém generátorů $V$.

</div>

V definici pod pojmem systém rozumíme uspořádanou množinu, časem uvidíme, proč je uspořádání důležité (pro souřadnice atp.). Nicméně pro jednoduchost značení budeme bázi, skládající se z konečně mnoha vektorů $v_1, \ldots, v_n$, značit $\lbrace v_1, \ldots, v_n \rbrace$.

Báze je tedy podle definice takový systém generátorů prostoru $V$, který je minimální ve smyslu inkluze. Každý z generátorů má svůj smysl, nemůžeme žádný vynechat, jinak bychom nevygenerovali celý prostor $V$.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 5.32 — Příklady bází)</span></p>

- V $\mathbb{R}^2$ máme bázi např. $e_1 = (1, 0)^T$, $e_2 = (0, 1)^T$. Jiná báze je $(7, 5)^T$, $(2, 3)^T$.
- V $\mathbb{R}^n$ máme např. bázi $e_1, \ldots, e_n$, říká se jí *kanonická* a značí se kan. Každý vektor $v = (v_1, \ldots, v_n)^T \in \mathbb{R}^n$ se dá vyjádřit jako lineární kombinace vektorů báze jednoduše jako $v = \sum_{i=1}^{n} v_i e_i$.
- V $\mathcal{P}^n$ je bází např. $1, x, x^2, \ldots, x^n$. Každý polynom $p \in \mathcal{P}^n$ v základním tvaru $p(x) = a_n x^n + \ldots + a_1 x + a_0$ již je vyjádřený jako lineární kombinace bázických vektorů (v opačném pořadí). Toto je na první pohled nejjednodušší báze, nikoliv však jediná možná. Bernsteinova báze se skládá z vektorů $\binom{n}{i}x^i(1 - x)^{n-i}$ pro $i = 0, 1, \ldots, n$ a používá se pro různé aproximace, např. ve výpočetní geometrii pro aproximaci křivek procházejících nebo kontrolovaných danými body (tzv. Bézierovy křivky, používají se třeba v typografii pro popis fontů).
- V $\mathcal{P}$ je bází např. nekonečný spočetný systém polynomů $1, x, x^2, \ldots$
- V prostoru $\mathcal{C}_{[a,b]}$ také existuje báze, ale není ji jednoduché žádnou explicitně vyjádřit.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Věta 5.33 — Jednoznačnost vyjádření v bázi)</span></p>

Nechť $v_1, \ldots, v_n$ je báze prostoru $V$. Pak pro každý vektor $u \in V$ existují jednoznačně určené koeficienty $\alpha_1, \ldots, \alpha_n \in \mathbb{T}$ takové, že $u = \sum_{i=1}^{n} \alpha_i v_i$.

*Důkaz.* Vektory $v_1, \ldots, v_n$ tvoří bázi $V$, tedy každé $u \in V$ se dá vyjádřit jako $u = \sum_{i=1}^{n} \alpha_i v_i$ pro vhodné skaláry $\alpha_1, \ldots, \alpha_n \in \mathbb{T}$. Jednoznačnost ukážeme sporem. Nechť existuje i jiné vyjádření $u = \sum_{i=1}^{n} \beta_i v_i$. Potom $\sum_{i=1}^{n} \alpha_i v_i - \sum_{i=1}^{n} \beta_i v_i = u - u = o$, neboli $\sum_{i=1}^{n} (\alpha_i - \beta_i)v_i = o$. Protože $v_1, \ldots, v_n$ jsou lineárně nezávislé, musí $\alpha_i = \beta_i$ pro každé $i = 1, \ldots, n$. To je spor s tím, že vyjádření jsou různá.

</div>

Díky zmíněné jednoznačnosti můžeme zavést pojem souřadnice.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Definice 5.34 — Souřadnice)</span></p>

Nechť $B = \lbrace v_1, \ldots, v_n \rbrace$ je báze prostoru $V$ a nechť vektor $u \in V$ má vyjádření $u = \sum_{i=1}^{n} \alpha_i v_i$. Pak *souřadnicemi* vektoru $u$ vzhledem k bázi $B$ rozumíme koeficienty $\alpha_1, \ldots, \alpha_n$ a vektor souřadnic značíme $[u]_B := (\alpha_1, \ldots, \alpha_n)^T$.

</div>

Pojem souřadnic je důležitější, než se na první pohled zdá. Umožňuje totiž reprezentovat těžko uchopitelné vektory a (konečně generované) prostory pomocí souřadnic, tedy aritmetických vektorů. Každý vektor má určité souřadnice a naopak každá $n$-tice skalárů dává souřadnici nějakého vektoru. Existuje tedy vzájemně jednoznačný vztah mezi vektory a souřadnicemi, který později (sekce 6.3) využijeme k tomu, abychom řadu např. početních, problémů z prostoru $V$ převedli do aritmetického prostoru, kde se pracuje snadněji.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 5.35 — Souřadnice vektoru vzhledem k bázi v $\mathbb{R}^2$)</span></p>

- Souřadnice vektoru $(-2, 3)^T$ vzhledem ke kanonické bázi: $[(-2, 3)^T]_{\text{kan}} = (-2, 3)^T$.
- Souřadnice vektoru $(-2, 3)^T$ vzhledem k bázi $B = \lbrace (-3, 1)^T, (1, 1)^T \rbrace$: $[(-2, 3)^T]_B = (\tfrac{5}{4}, \tfrac{7}{4})^T$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 5.36)</span></p>

Pro každé $v \in \mathbb{R}^n$ je $[v]_{\text{kan}} = v$, neboť vektor $v = (v_1, \ldots, v_n)^T$ má vyjádření $v = \sum_{i=1}^{n} v_i e_i$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 5.37)</span></p>

Uvažujme bázi $B = \lbrace 1, x, x^2 \rbrace$ prostoru $\mathcal{P}^2$. Pak $[3x^2 - 5]_B = (-5, 0, 3)^T$. Obecně každý polynom $p \in \mathcal{P}^n$ v základním tvaru $p(x) = a_n x^n + \ldots + a_1 x + a_0$ má vzhledem k bázi $B = \lbrace 1, x, x^2, \ldots, x^n \rbrace$ souřadnice $[p]_B = (a_0, a_1, \ldots, a_n)^T$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 5.38)</span></p>

Buď $B = \lbrace v_1, \ldots, v_n \rbrace$ báze prostoru $V$. Potom $[v_1]_B = (1, 0, \ldots, 0)^T = e_1$, $[v_2]_B = e_2$, $\ldots$, $[v_n]_B = e_n$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 5.39 — Pozorování)</span></p>

Nahlédněte následující pozorování pro vektorový prostor $V$:

- Je-li $v_1, \ldots, v_n \in V$ systém generátorů $V$, pak každý vektor $u \in V$ lze vyjádřit jako lineární kombinaci vektorů $v_1, \ldots, v_n$ alespoň jedním způsobem.
- Jsou-li $v_1, \ldots, v_n \in V$ lineárně nezávislé, pak každý vektor $u \in V$ lze vyjádřit jako lineární kombinaci vektorů $v_1, \ldots, v_n$ nejvýše jedním způsobem.
- Je-li $v_1, \ldots, v_n \in V$ báze $V$, pak každý vektor $u \in V$ lze vyjádřit jako lineární kombinaci vektorů $v_1, \ldots, v_n$ právě jedním způsobem.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Tvrzení 5.40 — Linearita souřadnic)</span></p>

Pro libovolnou bázi $B$ konečně generovaného prostoru $V$ nad $\mathbb{T}$, vektory $u, v \in V$ a skalár $\alpha \in \mathbb{T}$ platí

$$[u + v]_B = [u]_B + [v]_B, \qquad [\alpha v]_B = \alpha [v]_B.$$

*Důkaz.* Nechť báze $B$ sestává z vektorů $z_1, \ldots, z_n$, nechť $u = \sum_{i=1}^{n} \beta_i z_i$ a nechť $v = \sum_{i=1}^{n} \gamma_i z_i$. Potom $u + v = \sum_{i=1}^{n} (\beta_i + \gamma_i) z_i$ a tedy $[u + v]_B = (\beta_1 + \gamma_1, \ldots, \beta_n + \gamma_n)^T = [u]_B + [v]_B$. Podobně pro násobek $\alpha[u]_B = \alpha(\beta_1, \ldots, \beta_n)^T = (\alpha \beta_1, \ldots, \alpha \beta_n)^T = [\alpha u]_B$.

</div>

Vlastnost z tvrzení můžeme zobecnit: Souřadnice libovolné lineární kombinace vektorů jsou rovny té samé lineární kombinaci jejich souřadnic. Souřadnice tedy zachovávají jistou strukturu a vazby mezi vektory (lineární závislost aj.). Později v kapitole 6 uvidíme, že díky této vlastnosti dokážeme efektivně vyjadřovat souřadnice.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Věta 5.41 — O existenci báze)</span></p>

Každý vektorový prostor má bázi.

*Důkaz.* Důkaz provedeme pouze pro konečně generovaný prostor $V$. Buď $v_1, \ldots, v_n$ systém generátorů prostoru $V$. Jsou-li vektory lineárně nezávislé, tak už tvoří bázi. Jinak podle důsledku 5.29 existuje index $k$ tak, že $\operatorname{span}\lbrace v_1, \ldots, v_n \rbrace = \operatorname{span}\lbrace v_1, \ldots, v_{k-1}, v_{k+1}, \ldots, v_n \rbrace$. Tedy odstraněním $v_k$ bude systém vektorů stále generovat $V$. Je-li nyní systém vektorů lineárně nezávislý, tvoří bázi. Jinak postup opakujeme dokud nenajdeme bázi. Postup je konečný, protože máme konečnou množinu generátorů, tudíž bázi najít musíme.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(Lemma 5.42 — O výměně)</span></p>

Buď $y_1, \ldots, y_n$ systém generátorů vektorového prostoru $V$ a nechť vektor $x \in V$ má vyjádření $x = \sum_{i=1}^{n} \alpha_i y_i$. Pak pro libovolné $k$ takové, že $\alpha_k \neq 0$, je $y_1, \ldots, y_{k-1}, x, y_{k+1}, \ldots, y_n$ systém generátorů prostoru $V$.

*Důkaz.* Ze vztahu $x = \sum_{i=1}^{n} \alpha_i y_i$ vyjádříme $y_k = \tfrac{1}{\alpha_k}(x - \sum_{i \neq k} \alpha_i y_i)$. Chceme dokázat, že vektory $y_1, \ldots, y_{k-1}, x, y_{k+1}, \ldots, y_n$ generují prostor $V$. Vezměme libovolný vektor $z \in V$. Pro vhodné koeficienty $\beta_i$ můžeme vektor $z$ vyjádřit jako $z = \sum_{i=1}^{n} \beta_i y_i = \beta_k y_k + \sum_{i \neq k} \beta_i y_i = \tfrac{\beta_k}{\alpha_k}x + \sum_{i \neq k} (\beta_i - \tfrac{\beta_k}{\alpha_k}\alpha_i) y_i$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Věta 5.44 — Steinitzova věta o výměně)</span></p>

Buď $V$ vektorový prostor, buď $x_1, \ldots, x_m$ lineárně nezávislý systém ve $V$, a nechť $y_1, \ldots, y_n$ je systém generátorů $V$. Pak platí

1. $m \le n$,
2. existují navzájem různé indexy $k_1, \ldots, k_{n-m}$ takové, že $x_1, \ldots, x_m, y_{k_1}, \ldots, y_{k_{n-m}}$ tvoří systém generátorů $V$.

*Důkaz.* Důkaz provedeme matematickou indukcí podle $m$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(Důsledek 5.45)</span></p>

Všechny báze konečně generovaného vektorového prostoru $V$ jsou stejně velké.

*Důkaz.* Buďte $x_1, \ldots, x_m$ a $y_1, \ldots, y_n$ dvě báze prostoru $V$. Speciálně, $x_1, \ldots, x_m$ jsou lineárně nezávislé a $y_1, \ldots, y_n$ jsou generátory $V$, tedy $m \le n$. Analogicky naopak, $y_1, \ldots, y_n$ jsou lineárně nezávislé a $x_1, \ldots, x_m$ generují $V$, tedy $n \le m$. Dohromady dostáváme $m = n$.

</div>

### 5.5 Dimenze

Každý konečně generovaný prostor má bázi (věta 5.41) a všechny báze jsou stejně velké (důsledek 5.45), což ospravedlňuje zavedení dimenze prostoru jako velikosti (libovolné) báze.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Definice 5.46 — Dimenze)</span></p>

*Dimenze* konečně generovaného vektorového prostoru je velikost nějaké jeho báze. Dimenzi prostoru, který není konečně generovaný, je $\infty$. Dimenzi prostoru $V$ značíme $\dim V$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 5.47 — Příklady dimenzí)</span></p>

- $\dim \mathbb{R}^n = n$, $\dim \mathbb{R}^{m \times n} = mn$, $\dim \lbrace o \rbrace = 0$, $\dim \mathcal{P}^n = n + 1$,
- reálné prostory $\mathcal{P}$, $\mathcal{F}$, a prostor $\mathbb{R}$ nad $\mathbb{Q}$ nejsou konečně generované, mají dimenzi $\infty$ (viz problém 5.1).

</div>

Nadále budeme uvažovat pouze konečně generované vektorové prostory.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Tvrzení 5.48 — Vztah počtu prvků systému k dimenzi)</span></p>

Pro vektorový prostor $V$ platí:

1. Nechť $x_1, \ldots, x_m \in V$ jsou lineárně nezávislé. Pak $m \le \dim V$. Pokud $m = \dim V$, potom $x_1, \ldots, x_m$ je báze.
2. Nechť $y_1, \ldots, y_n$ jsou generátory $V$. Pak $n \ge \dim V$. Pokud $n = \dim V$, potom $y_1, \ldots, y_n$ je báze.

*Důkaz.* Označme $d = \dim V$ a nechť $z_1, \ldots, z_d$ je báze prostoru $V$, tedy jeho lineárně nezávislé generátory. (1) Protože $x_1, \ldots, x_m$ jsou lineárně nezávislé a $z_1, \ldots, z_d$ generátory $V$, tak podle Steinitzovy věty 5.44 je $m \le d$. Pokud $m = d$, pak podle stejné věty lze systém $x_1, \ldots, x_m$ doplnit o $d - m = 0$ vektorů na systém generátorů prostoru $V$. Tedy jsou to nutně generátory a tím i báze. (2) Protože $y_1, \ldots, y_n$ jsou generátory prostoru $V$ a $z_1, \ldots, z_d$ lineárně nezávislé, tak podle Steinitzovy věty 5.44 je $n \ge d$. Nechť $n = d$. Jsou-li $y_1, \ldots, y_n$ lineárně nezávislé, pak tvoří bázi. Pokud jsou lineárně závislé, pak lze jeden vynechat a získat systém generátorů o velikosti $n - 1$ (důsledek 5.29). Podle Steinitzovy věty by pak platilo $d \le n - 1$, což vede ke sporu.

</div>

První část tvrzení 5.48 mj. říká, že na bázi se dá nahlížet jako na maximální lineárně nezávislý systém. Druhá část věty pak říká, že báze je minimální systém generátorů (co do inkluze i co do počtu).

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Věta 5.49 — Rozšíření lineárně nezávislého systému na bázi)</span></p>

Každý lineárně nezávislý systém vektorového prostoru $V$ lze rozšířit na bázi $V$.

*Důkaz.* Nechť $x_1, \ldots, x_m$ jsou lineárně nezávislé a $z_1, \ldots, z_d$ je báze prostoru $V$. Podle Steinitzovy věty 5.44 existují indexy $k_1, \ldots, k_{d-m}$ takové, že $x_1, \ldots, x_m, z_{k_1}, \ldots, z_{k_{d-m}}$ jsou generátory $V$. Jejich počet je $d$, tedy podle tvrzení 5.48 je to i báze $V$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Věta 5.50 — Dimenze podprostoru)</span></p>

Je-li $W$ podprostorem prostoru $V$, pak $\dim W \le \dim V$. Pokud navíc $\dim W = \dim V$, tak $W = V$.

*Důkaz.* Definujme množinu $M := \emptyset$. Pokud $\operatorname{span}(M) = W$, jsme hotovi. V opačném případě existuje vektor $v \in W \setminus \operatorname{span}(M)$. Přidáme vektor $v$ do množiny $M$ a celý postup opakujeme. Protože $M$ je lineárně nezávislá množina vektorů, podle tvrzení 5.48 je velikost $M$ shora omezena dimenzí prostoru $V$. Proces je tedy konečný. Protože $\operatorname{span}(M) = W$, množina $M$ tvoří bázi prostoru $W$, a proto $\dim W \le \dim V$.

Pokud $\dim W = \dim V$, tak množina $M$ musí podle tvrzení 5.48 tvořit bázi $V$, a proto $W = V$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 5.51 — Podprostory $\mathbb{R}^2$)</span></p>

Najděme všechny podprostory prostoru $\mathbb{R}^2$:

- dimenze 2: to je pouze $\mathbb{R}^2$ (z věty 5.50),
- dimenze 1: ty jsou generovány jedním vektorem, tedy jsou to všechny přímky procházející počátkem,
- dimenze 0: to je pouze $\lbrace o \rbrace$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 5.52 — Struktura podprostorů)</span></p>

K tomu, abychom ilustrovali strukturu podprostorů, nejprve uvažujme všechny podmnožiny množiny $\lbrace 1, \ldots, n \rbrace$ a relaci „býti podmnožinou", neboli inkluzi $\subseteq$. Některé podmnožiny jsou neporovnatelné co do inkluze a jiné zase jsou. Inkluze je tedy částečné uspořádání a můžeme ji znázornit tzv. *Hasseovým diagramem*, kde spojnice značí „sousední" podmnožiny v inkluzi.

Podobným způsobem můžeme znázornit i strukturu podprostorů prostoru $V$ dimenze $n$, protože relace „býti podprostorem" je také částečné uspořádání. Diagram bude mít $n + 1$ hladin, přičemž na $i$-té hladině budou podprostory dimenze $i$. Ty jsou mezi sebou neporovnatelné ve smyslu inkluze či ve smyslu „býti podprostorem", nicméně některé vektory mohou sdílet.

</div>

#### Spojení podprostorů

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Definice 5.53 — Spojení podprostorů)</span></p>

Buďte $U, V$ podprostory vektorového prostoru $W$. Pak *spojení podprostorů* $U, V$ je definováno jako $U + V := \lbrace u + v ;\ u \in U, v \in V \rbrace$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Tvrzení 5.54)</span></p>

Buďte $U, V$ podprostory vektorového prostoru $W$. Pak

$$U + V = \operatorname{span}(U \cup V).$$

*Důkaz.* Inkluze „$\subseteq$": je triviální, neboť prostor $\operatorname{span}(U \cup V)$ je uzavřený na součty. Inkluze „$\supseteq$": Stačí ukázat, že $U + V$ obsahuje prostory $U, V$ a že je podprostorem $W$. První část je zřejmá, pro druhou uvažujme $x_1, x_2 \in U + V$. Vektory se dají vyjádřit jako $x_1 = u_1 + v_1$, $u_1 \in U$, $v_1 \in V$, a $x_2 = u_2 + v_2$, $u_2 \in U$, $v_2 \in V$. Potom $x_1 + x_2 = (u_1 + u_2) + (v_1 + v_2) \in U + V$, což dokazuje uzavřenost na sčítání. Pro uzavřenost na násobky uvažujme $x = u + v \in U + V$, $u \in U$, $v \in V$ a skalár $\alpha$. Pak $\alpha x = \alpha(u + v) = (\alpha u) + (\alpha v) \in U + V$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 5.55)</span></p>

- $\mathbb{R}^2 = \operatorname{span}\lbrace e_1 \rbrace + \operatorname{span}\lbrace e_2 \rbrace$,
- $\mathbb{R}^3 = \operatorname{span}\lbrace e_1 \rbrace + \operatorname{span}\lbrace e_2 \rbrace + \operatorname{span}\lbrace e_3 \rbrace$,
- $\mathbb{R}^3 = \operatorname{span}\lbrace e_1, e_2 \rbrace + \operatorname{span}\lbrace e_3 \rbrace$,
- $\mathbb{R}^2 = \operatorname{span}\lbrace (1, 2)^T \rbrace + \operatorname{span}\lbrace (3, 4)^T \rbrace$,
- ale i $\mathbb{R}^2 = \operatorname{span}\lbrace (1, 2)^T \rbrace + \operatorname{span}\lbrace (3, 4)^T \rbrace + \operatorname{span}\lbrace (5, 6)^T \rbrace$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Věta 5.56 — Dimenze spojení a průniku)</span></p>

Buďte $U, V$ podprostory vektorového prostoru $W$. Pak platí

$$\dim(U + V) + \dim(U \cap V) = \dim U + \dim V.$$

*Důkaz.* $U \cap V$ je podprostor prostoru $W$, tedy má konečnou bázi $z_1, \ldots, z_p$. Podle věty 5.49 ji můžeme rozšířit na bázi $U$ tvaru $z_1, \ldots, z_p, x_1, \ldots, x_m$. Podobně ji můžeme rozšířit na bázi $V$ tvaru $z_1, \ldots, z_p, y_1, \ldots, y_n$. Stačí, když ukážeme, že vektory $z_1, \ldots, z_p, x_1, \ldots, x_m, y_1, \ldots, y_n$ dohromady tvoří bázi $U + V$, a rovnost $(5.3)$ už bude platit.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Poznámka 5.58 — Direktní součet podprostorů)</span></p>

Je-li $U \cap V = \lbrace o \rbrace$, pak spojení podprostorů $W = U + V$ se nazývá *direktní součet* podprostorů $U, V$ a značí se $W = U \oplus V$. Podle věty 5.56 je $\dim(U \oplus V) = \dim U + \dim V$. Podmínka $U \cap V = \lbrace o \rbrace$ pak navíc způsobí, že každý vektor $w \in W$ lze zapsat jediným způsobem ve tvaru $w = u + v$, kde $u \in U$ a $v \in V$ (viz problém 5.2). Nyní jsou např. $\mathbb{R}^2 = \operatorname{span}\lbrace (1, 2)^T \rbrace \oplus \operatorname{span}\lbrace (3, 4)^T \rbrace$ nebo $\mathbb{R}^3 = \operatorname{span}\lbrace e_1 \rbrace \oplus \operatorname{span}\lbrace e_2 \rbrace \oplus \operatorname{span}\lbrace e_3 \rbrace$ direktními součty, ale $\mathbb{R}^2 = \operatorname{span}\lbrace (1, 2)^T \rbrace \oplus \operatorname{span}\lbrace (3, 4)^T \rbrace \oplus \operatorname{span}\lbrace (5, 6)^T \rbrace$ není.

</div>

### 5.6 Maticové prostory

Nyní skloubíme teorii matic s vektorovými prostory. Oba obory se vzájemně obohatí: Vektorově prostorový pohled nám umožní jednoduše odvodit další vlastnosti matic, a naopak, postupy z maticové teorie nám poskytnou nástroje na testování lineární nezávislosti, určování dimenze atp.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Definice 5.59 — Maticové prostory)</span></p>

Buď $A \in \mathbb{T}^{m \times n}$. Pak definujeme

1. sloupcový prostor $\mathcal{S}(A) := \operatorname{span}\lbrace A_{*1}, \ldots, A_{*n} \rbrace$,
2. řádkový prostor $\mathcal{R}(A) := \mathcal{S}(A^T)$,
3. jádro $\operatorname{Ker}(A) := \lbrace x \in \mathbb{T}^n ;\ Ax = o \rbrace$.

</div>

Sloupcový prostor je tedy prostor generovaný sloupci matice $A$, a je to podprostor $\mathbb{T}^m$. Podobně řádkový prostor je prostor generovaný řádky matice $A$, ale jedná se o podprostor $\mathbb{T}^n$. Jádro $\operatorname{Ker}(A)$ pak je tvořeno všemi řešeními soustavy $Ax = o$ a jedná se o také podprostor $\mathbb{T}^n$, neboť jsou splněny tři základní vlastnosti:

- Jádro obsahuje nulový vektor: $Ao = o$.
- Jádro je uzavřené na součty: Jsou-li $x, y \in \mathbb{T}^n$ řešeními soustavy, pak $Ax = o$, $Ay = o$. Součtem rovnic dostaneme $A(x + y) = o$, tedy i vektor $x + y$ náleží do jádra.
- Jádro je uzavřené na násobky: Je-li vektor $x \in \mathbb{T}^n$ řešením soustavy, pak $Ax = o$. Pro libovolné $\alpha \in \mathbb{T}$ platí $A(\alpha x) = \alpha(Ax) = \alpha o = o$, tedy i vektor $\alpha x$ náleží do jádra.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 5.60)</span></p>

Uvažme reálnou matici

$$A = \begin{pmatrix} 1 & 1 & 1 \\ 0 & 1 & 0 \end{pmatrix}.$$

Pak její sloupcový prostor je $\mathcal{S}(A) = \mathbb{R}^2$ a její řádkový prostor je $\mathcal{R}(A) = \operatorname{span}\lbrace (1, 1, 1)^T, (0, 1, 0)^T \rbrace$. Jádro matice $A$ určíme vyřešením soustavy $Ax = o$. Matice $A$ již je v odstupňovaném tvaru, proto pomocí volné proměnné $x_3$ popíšeme množinu řešení jako $\lbrace (x_3, 0, -x_3)^T ;\ x_3 \in \mathbb{R} \rbrace$. Jádro má tedy tvar $\operatorname{Ker}(A) = \operatorname{span}\lbrace (1, 0, -1)^T \rbrace$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Tvrzení 5.61)</span></p>

Buď $A \in \mathbb{T}^{m \times n}$. Pak

1. $\mathcal{S}(A) = \lbrace Ax ;\ x \in \mathbb{T}^n \rbrace$,
2. $\mathcal{R}(A) = \lbrace A^T y ;\ y \in \mathbb{T}^m \rbrace$.

*Důkaz.* Zřejmý z toho, že $Ax = \sum_{j=1}^{n} x_j A_{*j}$ představuje lineární kombinaci sloupců matice $A$. V druhé části analogicky $A^T y$ představuje lineární kombinaci řádků matice $A$.

</div>

Maticově můžeme reprezentovat libovolný podprostor $V$ prostoru $\mathbb{T}^n$. Stačí vzít nějaké jeho generátory $v_1, \ldots, v_m$ a sestavit matici $A \in \mathbb{T}^{m \times n}$, jejíž řádky tvoří právě vektory $v_1, \ldots, v_m$. Pak $V = \mathcal{R}(A)$. Podobně $V$ můžeme vyjádřit jako sloupcový prostor vhodné matice z $\mathbb{T}^{n \times m}$. Dokonce můžeme prostor $V$ reprezentovat i jako jádro vhodné matice z $\mathbb{T}^{m \times n}$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Tvrzení 5.62)</span></p>

Buď $V$ podprostor prostoru $\mathbb{T}^n$. Pak

1. $V = \mathcal{S}(A)$ pro vhodnou matici $A \in \mathbb{T}^{n \times m}$,
2. $V = \mathcal{R}(A)$ pro vhodnou matici $A \in \mathbb{T}^{m \times n}$,
3. $V = \operatorname{Ker}(A)$ pro vhodnou matici $A \in \mathbb{T}^{m \times n}$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Poznámka 5.63 — Geometrický pohled na maticové prostory)</span></p>

Uvažujme zobrazení $x \mapsto Ax$ s maticí $A \in \mathbb{T}^{m \times n}$. Jádro matice $A$ je tedy tvořeno všemi vektory z $\mathbb{T}^n$, které se zobrazí na nulový vektor. Sloupcový prostor $\mathcal{S}(A)$ matice $A$ pak zase představuje množinu všech obrazů, neboli obraz prostoru $\mathbb{T}^n$ při tomto zobrazení. Jak později ukážeme, tyto prostory hrají klíčovou roli pro analýzu geometrické struktury tohoto zobrazení.

</div>

#### Prostory a násobení maticí zleva

Podívejme se, jak se mění maticové prostory, když matici násobíme zleva nějakou jinou maticí (to vlastně dělá Gaussova eliminace).

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Tvrzení 5.64 — Prostory a násobení maticí zleva)</span></p>

Buď $A \in \mathbb{T}^{m \times n}$, $Q \in \mathbb{T}^{p \times m}$. Pak

1. $\mathcal{R}(QA)$ je podprostorem $\mathcal{R}(A)$,
2. Pokud $A_{*k} = \sum_{j \neq k} \alpha_j A_{*j}$ pro nějaké $k \in \lbrace 1, \ldots, n \rbrace$ a nějaká $\alpha_j \in \mathbb{T}$, $j \neq k$, pak $(QA)_{*k} = \sum_{j \neq k} \alpha_j (QA)_{*j}$.

*Důkaz.* (1) Stačí ukázat $\mathcal{R}(QA) \subseteq \mathcal{R}(A)$. Buď $x \in \mathcal{R}(QA)$, pak existuje $y \in \mathbb{T}^p$ takové, že $x = (QA)^T y = A^T(Q^T y) \in \mathcal{R}(A)$. (2) $(QA)_{*k} = QA_{*k} = Q(\sum_{j \neq k} \alpha_j A_{*j}) = \sum_{j \neq k} \alpha_j QA_{*j} = \sum_{j \neq k} \alpha_j (QA)_{*j}$.

</div>

Věta říká, že řádkové prostory jsou porovnatelné přímo — po pronásobení libovolnou maticí zleva dostaneme podprostor. To se snadno nahlédne i z toho, že každý řádek matice $QA$ je vlastně lineární kombinací řádků matice $A$ (viz poznámka 5.21), a vybranými lineárními kombinacemi lze vygenerovat pouze podprostor. Také lineárně závislostní vazba: je-li $i$-tý sloupec matice $A$ závislý na ostatních, potom $i$-tý sloupec matice $QA$ je závislý na ostatních se stejnou lineární kombinací (pozor, lineární nezávislost se nemusí zachovávat).

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Tvrzení 5.66 — Prostory a násobení regulární maticí zleva)</span></p>

Buď $Q \in \mathbb{T}^{m \times m}$ regulární a $A \in \mathbb{T}^{m \times n}$. Pak

1. $\mathcal{R}(QA) = \mathcal{R}(A)$,
2. Rovnost $A_{*k} = \sum_{j \neq k} \alpha_j A_{*j}$ platí právě tehdy, když $(QA)_{*k} = \sum_{j \neq k} \alpha_j (QA)_{*j}$, kde $k \in \lbrace 1, \ldots, n \rbrace$ a $\alpha_j \in \mathbb{T}$, $j \neq k$.

*Důkaz.* (1) Podle tvrzení 5.64 je $\mathcal{R}(QA) \subseteq \mathcal{R}(A)$. Aplikujeme-li tvrzení 5.64 na matici $(QA)$ násobenou zleva $Q^{-1}$, tak dostaneme $\mathcal{R}(Q^{-1}QA) \subseteq \mathcal{R}(QA)$, tedy $\mathcal{R}(A) \subseteq \mathcal{R}(QA)$. Dohromady máme $\mathcal{R}(QA) = \mathcal{R}(A)$. (2) Implikaci zleva doprava dostaneme z tvrzení 5.64. Obrácenou implikaci dostaneme z tvrzení 5.64 aplikovaného na matici $(QA)$ násobenou zleva $Q^{-1}$.

</div>

Důsledkem předchozí věty je, že pokud některé sloupce matice $A$ jsou lineárně nezávislé, tak zůstanou i po vynásobení regulární maticí zleva.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 5.65)</span></p>

V matici $A$ je druhý sloupeček dvojnásobkem prvního a tato vlastnost zůstane i pro výsledný součin $QA$:

$$QA = \begin{pmatrix} 1 & 2 & -1 \\ -2 & 1 & 1 \end{pmatrix} \begin{pmatrix} 1 & 2 & 4 \\ 2 & 4 & 5 \\ 1 & 2 & 7 \end{pmatrix} = \begin{pmatrix} 4 & 8 & 7 \\ 1 & 2 & 4 \end{pmatrix}.$$

V matici $A'$ je třetí sloupeček součtem prvních dvou a tato vlastnost opět zůstane i pro výsledný součin $QA'$:

$$QA' = \begin{pmatrix} 1 & 2 & -1 \\ -2 & 1 & 1 \end{pmatrix} \begin{pmatrix} 1 & 1 & 2 \\ 1 & 2 & 3 \\ 1 & 3 & 4 \end{pmatrix} = \begin{pmatrix} 2 & 2 & 4 \\ 0 & 3 & 3 \end{pmatrix}.$$

</div>

#### Maticové prostory a RREF

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Věta 5.68 — Maticové prostory a RREF)</span></p>

Buď $A \in \mathbb{T}^{m \times n}$ a buď $A^R$ její RREF tvar s pivoty na pozicích $(1, p_1), \ldots, (r, p_r)$, kde $r = \operatorname{rank}(A)$. Pak

1. nenulové řádky $A^R$, tedy vektory $A^R_{1*}, \ldots, A^R_{r*}$, tvoří bázi $\mathcal{R}(A)$,
2. sloupce $A_{*p_1}, \ldots, A_{*p_r}$ tvoří bázi $\mathcal{S}(A)$,
3. $\dim \mathcal{R}(A) = \dim \mathcal{S}(A) = r$.

*Důkaz.* Víme z věty 3.31, že $A^R = QA$ pro nějakou regulární matici $Q$. (1) Podle tvrzení 5.66 je $\mathcal{R}(A) = \mathcal{R}(QA) = \mathcal{R}(A^R)$. Nenulové řádky $A^R$ jsou lineárně nezávislé, tedy tvoří bázi $\mathcal{R}(A^R)$ i $\mathcal{R}(A)$. (2) Vektory $A^R_{*p_1}, \ldots, A^R_{*p_r}$ tvoří bázi $\mathcal{S}(A^R)$. Tyto vektory jsou jistě lineárně nezávislé (jsou to jednotkové vektory). Generují $\mathcal{S}(A^R)$, neboť libovolný nebázický sloupec se dá vyjádřit jako lineární kombinace těch bázických: $A^R_{*j} = \sum_{i=1}^{r} a^R_{ij} e_i = \sum_{i=1}^{r} a^R_{ij} A^R_{*p_i}$. Nyní použijeme tvrzení 5.66, která zaručí, že i $A_{*p_1}, \ldots, A_{*p_r}$ jsou lineárně nezávislé a generují $\mathcal{S}(A)$, tedy tvoří bázi $\mathcal{S}(A)$. (3) Hodnota $\dim \mathcal{R}(A)$ je velikost báze $\mathcal{R}(A)$, tedy $r$, a podobně $\dim \mathcal{S}(A)$ je také $r$. Navíc $r = \operatorname{rank}(A)$.

</div>

Zdůrazněme, že bázi řádkového prostoru $\mathcal{R}(A)$ najdeme v řádcích matice $A^R$, zatímco bázi sloupcového prostoru $\mathcal{S}(A)$ najdeme ve sloupcích původní matice $A$.

Třetí vlastnost věty 5.68 dává velmi netriviální důsledek pro hodnost matice a její transpozice, neboť

$$\operatorname{rank}(A) = \dim \mathcal{R}(A) = \dim \mathcal{S}(A) = \dim \mathcal{R}(A^T) = \operatorname{rank}(A^T).$$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Věta 5.69)</span></p>

Pro každou matici $A \in \mathbb{T}^{m \times n}$ platí $\operatorname{rank}(A) = \operatorname{rank}(A^T)$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 5.70)</span></p>

Uvažujme prostor $V = \operatorname{span}\lbrace (1, 2, 3, 4, 5)^T, (1, 1, 1, 1, 1)^T, (1, 3, 5, 7, 9)^T, (2, 1, 1, 0, 0)^T \rbrace \le \mathbb{R}^5$. Nejprve sestavme matici $A$, jejíž sloupce jsou rovny daným generátorům $V$, tedy $V = \mathcal{S}(A)$, a upravme ji na redukovaný odstupňovaný tvar:

$$A = \begin{pmatrix} 1 & 1 & 1 & 2 \\ 2 & 1 & 3 & 1 \\ 3 & 1 & 5 & 1 \\ 4 & 1 & 7 & 0 \\ 5 & 1 & 9 & 0 \end{pmatrix} \xrightarrow{\text{RREF}} \begin{pmatrix} 1 & 0 & 2 & 0 \\ 0 & 1 & -1 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \end{pmatrix}.$$

Z RREF tvaru vidíme, že $\dim(V) = \operatorname{rank}(A) = 3$ a báze $V$ je například $(1, 2, 3, 4, 5)^T$, $(1, 1, 1, 1, 1)^T$, $(2, 1, 1, 0, 0)^T$. Třetí z generátorů je závislý na ostatních, konkrétně je roven dvojnásobku prvního minus druhý (koeficienty vidíme ve třetím sloupci matice v RREF tvaru).

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Poznámka 5.71)</span></p>

Uvažujme soustavu lineárních rovnic $Ax = b$. Řešitelnost soustavy vlastně znamená, že vektor pravých stran $b$ se dá vyjádřit jako lineární kombinace sloupců matice $A$ (srov. poznámka 5.20). Tudíž soustava je řešitelná právě tehdy, když $b \in \mathcal{S}(A)$, neboli $\mathcal{S}(A) = \mathcal{S}(A \mid b)$. Věta 5.68 pak přímo dává znění Frobeniovy věty z poznámky 2.25.

</div>

#### Dimenze jádra a hodnost matice

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Věta 5.72 — O dimenzi jádra a hodnosti matice)</span></p>

Pro každou matici $A \in \mathbb{T}^{m \times n}$ platí

$$\dim \operatorname{Ker}(A) + \operatorname{rank}(A) = n.$$

*Důkaz.* Buď $\dim \operatorname{Ker}(A) = k$. Nechť vektory $v_1, \ldots, v_k$ tvoří bázi $\operatorname{Ker}(A)$, což mj. znamená, že $Av_1 = \ldots = Av_k = o$. Rozšíříme vektory $v_1, \ldots, v_k$ na bázi celého prostoru $\mathbb{T}^n$ doplněním o vektory $v_{k+1}, \ldots, v_n$. Stačí ukázat, že vektory $Av_{k+1}, \ldots, Av_n$ tvoří bázi $\mathcal{S}(A)$, protože pak $\operatorname{rank}(A) = \dim \mathcal{S}(A) = n - k$ a rovnost z věty je splněna.

„Generujícnost." Buď $y \in \mathcal{S}(A)$, pak $y = Ax$ pro nějaké $x \in \mathbb{T}^n$. Toto $x$ lze vyjádřit $x = \sum_{i=1}^{n} \alpha_i v_i$. Dosazením $y = Ax = A(\sum_{i=1}^{n} \alpha_i v_i) = \sum_{i=1}^{n} \alpha_i A v_i = \sum_{i=k+1}^{n} \alpha_i (Av_i)$.

„Lineární nezávislost." Buď $\sum_{i=k+1}^{n} \alpha_i Av_i = o$. Pak platí $A(\sum_{i=k+1}^{n} \alpha_i v_i) = o$, čili $\sum_{i=k+1}^{n} \alpha_i v_i$ patří do jádra matice $A$. Proto $\sum_{i=k+1}^{n} \alpha_i v_i = \sum_{i=1}^{k} \beta_i v_i$ pro nějaké skaláry $\beta_1, \ldots, \beta_k$. Přepsáním rovnice $\sum_{i=k+1}^{n} \alpha_i v_i + \sum_{i=1}^{k} (-\beta_i) v_i = o$ a vzhledem k lineární nezávislosti vektorů $v_1, \ldots, v_n$ je $\alpha_{k+1} = \ldots = \alpha_n = \beta_1 = \ldots = \beta_k = 0$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Poznámka 5.73 — Geometrický pohled na větu 5.72)</span></p>

Uvažujme zobrazení $x \mapsto Ax$ s maticí $A \in \mathbb{T}^{m \times n}$, viz poznámka 5.63. Prostor $\mathbb{T}^n$ se zobrazí na prostor $\mathcal{S}(A)$, jehož dimenze je $r = \operatorname{rank}(A)$. Tudíž zobrazení zobrazuje $n$-dimenzionální prostor na $r$-dimenzionální prostor. Právě ten deficit $n - r \ge 0$ je podle vzorečku (5.4) roven dimenzi jádra matice $A$. Pro regulární matici je jádro triviální ($\operatorname{Ker}(A) = \lbrace o \rbrace$), a proto zobrazuje $\mathbb{T}^n$ na celé $\mathbb{T}^n$. Čím je však jádro větší, tím menší je obraz prostoru $\mathbb{T}^n$. Dimenze jádra tedy popisuje míru „degenerace" zobrazení.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 5.74 — Výpočet báze jádra)</span></p>

Uvažujme matici a její RREF tvar

$$A = \begin{pmatrix} 2 & 4 & 4 & 4 \\ -3 & -4 & 2 & 0 \\ 5 & 7 & -2 & 1 \end{pmatrix} \xrightarrow{\text{RREF}} \begin{pmatrix} 1 & 0 & -6 & -4 \\ 0 & 1 & 4 & 3 \\ 0 & 0 & 0 & 0 \end{pmatrix}.$$

Tedy $\dim \operatorname{Ker}(A) = 4 - 2 = 2$. Prostor $\operatorname{Ker}(A)$ představuje všechna řešení soustavy $Ax = o$ a ta jsou tvaru $(6x_3 + 4x_4, -4x_3 - 3x_4, x_3, x_4)^T$, $x_3, x_4 \in \mathbb{R}$, neboli

$$x_3(6, -4, 1, 0)^T + x_4(4, -3, 0, 1)^T, \quad x_3, x_4 \in \mathbb{R}.$$

Tudíž vektory $(6, -4, 1, 0)^T$, $(4, -3, 0, 1)^T$ tvoří bázi $\operatorname{Ker}(A)$. Tyto vektory nalezneme i přímo tak, že za jednu nebázickou proměnnou dosadíme $1$, za zbylé nuly a dopočítáme hodnoty bázických proměnných. Tento postup platí univerzálně pro každou matici.

</div>

### 5.7 Aplikace

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 5.75 — Ještě ke kódování)</span></p>

Navažme na příklad 4.42 o Hammingově kódu $(7, 4, 3)$. Ke kódování jsme používali generující matici $H$ rozměru $7 \times 4$ jednoduše tak, že vstupní úsek $a$ délky $4$ se zakóduje na úsek $b := Ha$ délky $7$. Všechny zakódované úseky tak představují sloupcový prostor matice $H$. Protože $H$ má lineárně nezávislé sloupce, jedná se o podprostor dimenze $4$ v prostoru $\mathbb{Z}_2^7$.

Detekce chyb přijatého úseku $b$ probíhá pomocí detekční matice $D$ rozměru $3 \times 7$. Pokud $Db = o$, nenastala chyba (nebo nastaly alespoň dvě). Po detekční matici tedy chceme, aby (pouze) vektory ze sloupcového prostoru matice $H$ zobrazovala na nulový vektor. Tudíž musí $\mathcal{S}(H) = \operatorname{Ker}(D)$. Nyní již vidíme, proč má matice $D$ dané rozměry — aby její jádro byl čtyřdimenzionální podprostor, musí mít podle věty 5.72 hodnost $3$, a proto $3$ lineárně nezávislé řádky postačují.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 5.76 — Rozpoznávání obličejů)</span></p>

Detekce a rozpoznávání obličejů z digitálního obrazu je moderní úloha počítačové grafiky. Digitální obraz reprezentujeme jako matici $A \in \mathbb{R}^{m \times n}$, kde $a_{ij}$ udává barvu pixelu na pozici $i, j$. Množinu obrázků s obličeji si můžeme s jistou mírou zjednodušení představit jako podprostor prostoru všech obrázků $\mathbb{R}^{m \times n}$. Báze tohoto podprostoru jsou tzv. *eigenfaces*, čili určité základní typy nebo rysy obličeje, ze kterých skládáme ostatní obličeje.

Pokud chceme rozhodnout, zda obrázek odpovídá obličeji, tak spočítáme, zda odpovídající vektor leží v podprostoru obličejů nebo v jeho blízkosti. Podobně postupujeme, pokud chceme rozpoznat zda daný obrázek odpovídá nějakému známému obličeji: Ve vektorovém prostoru $\mathbb{R}^{m \times n}$ zjistíme, který z vektorů odpovídajících známým tvářím je nejblíže vektoru našeho obrázku.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 5.77 — Lagrangeův interpolační polynom)</span></p>

Vraťme se nyní k problému interpolace bodů polynomem. Mějme v rovině $n + 1$ bodů $(x_0, y_0), (x_1, y_1), \ldots, (x_n, y_n)$, kde $x_i \neq x_j$ pro $i \neq j$. Úkolem je najít polynom $p(x)$ procházející těmito body.

Polynomy $1, x, x^2, \ldots, x^n$ tvoří standardní bázi vektorového prostoru $\mathcal{P}^n$, a naším cílem vlastně je najít souřadnice $a_0, a_1, \ldots, a_n$ hledaného polynomu $p(x)$ vzhledem k této bázi.

Nyní se nabízí otázka, jestli bychom nenašli polynom snadněji, kdybychom zvolili jinou bázi prostoru $\mathcal{P}^n$? Odpověď zní „ano". Zvolíme následující bázi prostoru $\mathcal{P}^n$. Pro $i = 0, 1, \ldots, n$ definujeme polynom

$$p_i(x) = \prod_{j=0, j \neq i}^{n} \frac{1}{x_i - x_j}(x - x_j).$$

Tento polynom má v bodě $x_i$ hodnotu $1$ a v ostatních bodech $x_j$, $j \neq i$, hodnotu $0$. Je snadné nahlédnout, že tyto polynomy jsou lineárně nezávislé: žádný polynom $p_i(x)$ není lineární kombinací ostatních, protože ostatní polynomy mají v bodě $x_i$ hodnotu $0$. Tudíž polynomy $p_0(x), \ldots, p_n(x)$ tvoří bázi prostoru $\mathcal{P}^n$. Interpolační polynom $p(x)$ se tak dá jednoznačně vyjádřit jako lineární kombinace a souřadnice tvoří právě funkční hodnoty $y_0, \ldots, y_n$. Tím dostáváme explicitní vyjádření interpolačního polynomu v tzv. Lagrangeově tvaru

$$p(x) = \sum_{i=0}^{n} y_i p_i(x).$$

</div>

### Shrnutí ke kapitole 5

Vektorové prostory představují další abstraktní pojem. Vektory v prostoru umíme sčítat a každý jednotlivě násobit skalárem (nikoli nutně mezi sebou!). Aplikací obou operací na $n$ vektorů získáme lineární kombinaci těchto vektorů. Množina všech lineárních kombinací zadaných vektorů vytvoří vektorový podprostor. Pokud stejný podprostor nevygeneruje žádná ostře menší podmnožina vektorů, jsou tyto vektory lineárně nezávislé, jinak jsou lineárně závislé. Alternativně, vektory jsou závislé pokud mezi nimi je aspoň jeden, který je lineární kombinací ostatních. Lineárně nezávislé generátory prostoru se nazývají bází tohoto prostoru. Každý prostor má nějakou bázi a pokud jich je více, tak mají všechny stejnou velikost (Steinitzova věta o výměně). To nás opravňuje zavést dimenzi prostoru jako počet vektorů v bázi. Báze prostoru pak představuje jakýsi souřadný systém v tomto prostoru, protože každý vektor prostoru se dá jednoznačně vyjádřit jako lineární kombinace bázických vektorů; příslušné koeficienty se nazývají souřadnice.

Prostory úzce souvisí s maticemi, a to dvojím způsobem. S každou maticí $A$ je spjato několik vektorových prostorů: ten, generovaný sloupci, ten, generovaný řádky, a pak jádro, čili prostor řešení soustavy $Ax = o$. Tím, že jsme prozkoumali, jak elementární aj. maticové úpravy mění tyto prostory pak na druhou stranu dokážeme pomocí matic snadno řešit spoustu úloh: zjistit, zda dané vektory jsou lineárně nezávislé, určit dimenzi prostoru, který generují, vybrat z nich vhodnou bázi, spočítat souřadnice vektoru v dané bázi atp.

---

## Kapitola 6 — Lineární zobrazení

S lineárními zobrazeními jsme se již letmo setkali jako se zobrazeními typu $x \mapsto Ax$, kde $A \in \mathbb{T}^{m \times n}$. Nahlédli jsme, že zobrazení je bijekcí právě pro regulární matice a inverzní zobrazení má popis $y \mapsto A^{-1}y$. Dále víme, že prostor $\mathbb{T}^n$ se zobrazí na prostor $\mathcal{S}(A)$, jehož dimenze je $r = \operatorname{rank}(A)$. Rozdíl dimenzí $n - r$ vzoru a obrazu pak odpovídá dimenzi jádra matice $A$.

Pro lineární zobrazení $x \mapsto Ax$ zřejmě také platí

$$(x + y) \mapsto A(x + y) = Ax + Ay, \qquad (\alpha x) \mapsto A(\alpha x) = \alpha(Ax).$$

Právě tuto vlastnost použijeme jako definici lineárního zobrazení pro obecné prostory. Jinými slovy tato vlastnost říká, že obraz součtu dvou vektorů je roven součtu jejich obrazů a analogicky pro násobky. Tím pádem obraz lineární kombinace vektorů se dá vyjádřit jako lineární kombinace jejich obrazů. Lineární zobrazení tedy zachovává vztah mezi vektory: lineárně závislé vektory se zobrazí na lineárně závislé obrazy (ale ne naopak!); vektor, který je závislý na jiných vektorech se zobrazí na vektor závislý na jejich obrazech při stejné lineární kombinaci atp.

V celé kapitole uvažujeme pouze konečně generované vektorové prostory.

### 6.1 Lineární zobrazení mezi obecnými prostory

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Definice 6.1 — Lineární zobrazení)</span></p>

Buďte $U, V$ vektorové prostory nad tělesem $\mathbb{T}$. Zobrazení $f \colon U \to V$ je *lineární*, pokud pro každé $x, y \in U$ a $\alpha \in \mathbb{T}$ platí:

- $f(x + y) = f(x) + f(y)$,
- $f(\alpha x) = \alpha f(x)$.

</div>

Lineární zobrazení se též nazývá *homomorfismus*. Pro injektivní zobrazení je *monomorfismus*, surjektivní homomorfismus je *epimorfismus*, homomorfismus množiny do sebe sama je *endomorfismus*, surjektivní a injektivní homomorfismus je *isomorfismus*, a isomorfní endomorfismus se nazývá *automorfismus*.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 6.2 — Příklady lineárních zobrazení v rovině)</span></p>

Již v příkladu 3.21 jsme ukázali několik lineárních zobrazení daných předpisem $x \mapsto Ax$, kde $A \in \mathbb{R}^{2 \times 2}$. Tato zobrazení představovala různé transformace v rovině, konkrétně překlopení podle osy, natáhnutí podle osy a otočení kolem počátku. Projekci jako lineární zobrazení jsme uvedli v poznámce 3.43.

Lineární zobrazení s maticí $A = \binom{v_1 \ 0}{0 \ v_2}$ představuje škálování, které natahuje $v_1$-krát ve směru osy $x_1$ a $v_2$-krát ve směru osy $x_2$. Konkrétně pro hodnotu $v = (0.6, 0.6)^T$ dostaneme zobrazení, které rovnoměrně zmenšuje objekty.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 6.3 — Matice rotace)</span></p>

Odvodíme vyjádření lineárního zobrazení, které reprezentuje otočení v rovině kolem počátku o úhel $\alpha$ proti směru hodinových ručiček. Bod $(x_1, x_2)^T \in \mathbb{R}^2$ ztotožníme s komplexním číslem $z := x_1 + ix_2$ a označíme komplexní číslo $r := \cos(\alpha) + i\sin(\alpha)$. Jak víme ze sekce 1.4, násobení číslem $r$ reprezentuje otočení o úhel $\alpha$. Tudíž komplexní číslo $z$ se otočí na komplexní číslo

$$r \cdot z = (\cos(\alpha) + i\sin(\alpha)) \cdot (x_1 + ix_2) = \cos(\alpha)x_1 - \sin(\alpha)x_2 + i(\sin(\alpha)x_1 + \cos(\alpha)x_2).$$

Pokud zpátky ztotožníme komplexní čísla s body v rovině, tak dostáváme, že bod $(x_1, x_2)^T$ se zobrazí na bod $(\cos(\alpha)x_1 - \sin(\alpha)x_2, \ \sin(\alpha)x_1 + \cos(\alpha)x_2)^T$. Tudíž otočení tvoří lineární zobrazení a jeho maticové vyjádření je $x \mapsto Ax$, kde

$$A = \begin{pmatrix} \cos(\alpha) & -\sin(\alpha) \\ \sin(\alpha) & \cos(\alpha) \end{pmatrix}.$$

Speciálně, vektor $e_1 = (1, 0)^T$ se zobrazí na $(\cos(\alpha), \sin(\alpha))^T$ a vektor $e_2 = (0,1)^T$ se zobrazí na $(-\sin(\alpha), \cos(\alpha))^T$.

Konkrétně matice otočení o $90°$ a matice otočení o $180°$ mají tvar

$$\begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix} \quad \text{a} \quad \begin{pmatrix} -1 & 0 \\ 0 & -1 \end{pmatrix}.$$

Matici rotace snadno zobecníme na případ rotace v prostoru $\mathbb{R}^n$, pokud se omezíme pouze na otočení o úhel $\alpha$ v rovině os $x_i, x_j$. Schematicky (prázdné místo odpovídá nulám):

$$\begin{pmatrix} I & & & \\ & \cos(\alpha) & & -\sin(\alpha) & \\ & & I & & \\ & \sin(\alpha) & & \cos(\alpha) & \\ & & & & I \end{pmatrix}.$$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 6.4 — Další příklady lineárních zobrazení)</span></p>

- Typickým příkladem lineárního zobrazení je $f \colon \mathbb{R}^n \to \mathbb{R}^m$ definované $f(x) = Ax$, kde $A \in \mathbb{R}^{m \times n}$ je pevná matice. Jak uvidíme později v důsledku 6.20, tak žádné jiné lineární zobrazení mezi prostory $\mathbb{R}^n$ a $\mathbb{R}^m$ neexistuje.
- Triviální zobrazení $f \colon U \to V$ definované $f(x) = o$ je zjevně lineární.
- Identita $id \colon U \to U$ definovaná $id(x) = x$ je dalším příkladem lineárního zobrazení.
- Zobrazení $f \colon \mathbb{T}^{m \times n} \to \mathbb{T}^{n \times m}$ dané předpisem $f(A) = A^T$ je lineární díky vlastnostem maticové transpozice (tvrzení 3.13).
- Derivace z prostoru reálných diferencovatelných funkcí do prostoru reálných funkcí $\mathcal{F}$ představuje také lineární zobrazení, protože splňuje vlastnosti $(f + g)' = f' + g'$ a $(\alpha f)' = \alpha f'$ pro každé dvě funkce $f, g$ a skalár $\alpha \in \mathbb{R}$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Tvrzení 6.5 — Vlastnosti lineárních zobrazení)</span></p>

Buď $f \colon U \to V$ lineární zobrazení. Pak

1. $f\!\left(\sum_{i=1}^{n} \alpha_i x_i\right) = \sum_{i=1}^{n} \alpha_i f(x_i)$ pro každé $\alpha_i \in \mathbb{T}$, $x_i \in U$, $i = 1, \ldots, n$,
2. $f(o) = o$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Poznámka 6.6 — Geometrická vlastnost)</span></p>

Jedna z geometrických vlastností lineárních zobrazení je ta, že zobrazují přímku na přímku nebo na bod. Přímka (viz str. 120) určená dvěma různými vektory $v_1, v_2$ je množina vektorů tvaru $\lambda v_1 + (1 - \lambda)v_2$, kde $\lambda \in \mathbb{T}$. Obrazem této množiny při lineárním zobrazení $f$ je množina popsaná $f(\lambda v_1 + (1 - \lambda)v_2) = \lambda f(v_1) + (1 - \lambda)f(v_2)$, což je opět přímka nebo bod (je-li $f(v_1) = f(v_2)$). Pozor, opačným směrem tvrzení neplatí, ne každé zobrazení zachovávající přímky je lineární. Například posunutí je nelineární, ale zobrazuje přímky na přímky.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Definice 6.7 — Obraz a jádro)</span></p>

Buď $f \colon U \to V$ lineární zobrazení. Pak definujeme

- **obraz** $f(U) := \lbrace f(x);\ x \in U \rbrace$,
- **jádro** $\operatorname{Ker}(f) := \lbrace x \in U;\ f(x) = o \rbrace$.

Obraz má přirozený význam jako obor hodnot zobrazení. Definici můžeme rozšířit na obraz jakékoli podmnožiny $M \subseteq U$ takto: $f(M) := \lbrace f(x);\ x \in M \rbrace$.

</div>

Jádro popisuje určité rysy lineárního zobrazení. Triviální jádro (tj. $\operatorname{Ker}(f) = \lbrace o \rbrace$) značí, že zobrazení je prosté a tím pádem dimenze vzoru $U$ i obrazu $f(U)$ jsou stejné. Naopak, čím větší je jádro, tím více zobrazení degeneruje, více vektorů se zobrazí na tu samou hodnotu, a tím menší má obraz $f(U)$ dimenzi vzhledem k dimenzi vzoru $U$ (viz důsledek 6.43).

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Poznámka 6.8)</span></p>

Jádro matice a jádro lineárního zobrazení spolu úzce souvisí. Definujeme-li zobrazení $f$ předpisem $f(x) = Ax$, potom $\operatorname{Ker}(f) = \operatorname{Ker}(A)$ a $f(U) = \mathcal{S}(A)$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 6.9 — Obraz a jádro)</span></p>

Uvažujme lineární zobrazení $x \mapsto Ax$, kde $A \in \mathbb{R}^{2 \times 2}$, viz příklad 6.2.

- Pro matici $A = \binom{-1 \ 0}{\ 0 \ 1}$ představuje zobrazení překlopení podle osy $x_2$. Obraz je $f(\mathbb{R}^2) = \mathbb{R}^2$ a jádro je $\operatorname{Ker}(f) = \lbrace o \rbrace$.
- Pro matici $A = \binom{1 \ 0}{0 \ 0}$ dostáváme projekci na osu $x_1$. Obraz je nyní $f(\mathbb{R}^2) = \operatorname{span}\lbrace (1, 0)^T \rbrace$, tedy osa $x_1$, a jádro je $\operatorname{Ker}(f) = \operatorname{span}\lbrace (0, 1)^T \rbrace$, tedy osa $x_2$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Tvrzení 6.10)</span></p>

Buď $f \colon U \to V$ lineární zobrazení. Pak:

1. $f(U)$ je podprostorem $V$,
2. $\operatorname{Ker}(f)$ je podprostorem $U$,
3. pro každé $x_1, \ldots, x_n \in U$ platí: $f(\operatorname{span}\lbrace x_1, \ldots, x_n \rbrace) = \operatorname{span}\lbrace f(x_1), \ldots, f(x_n) \rbrace$.

</div>

Bod (3) tvrzení 6.10 zároveň dává návod jak určovat obraz podprostoru $W$ prostoru $U$: určíme obrazy báze (nebo obecně generátorů $W$), a ty tvoří generátory obrazu $f(W)$.

Připomeňme dva druhy zobrazení, prosté a „na". Lineární zobrazení $f \colon U \to V$ je „na", pokud $f(U) = V$. Jinými slovy, pro každý vektor $y \in V$ existuje vektor $x \in U$, který se na něj zobrazí, tj. $f(x) = y$. Rozhodnout, zda je zobrazení $f$ „na", lze snadno podle bodu (3) tvrzení 6.10. Stačí zvolit generátory prostoru $U$ a ověřit, jestli jejich obrazy generují prostor $V$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(Důsledek 6.11)</span></p>

Lineární zobrazení $f \colon U \to V$ je „na" právě tehdy, když se nějaké generátory prostoru $U$ zobrazí na generátory prostoru $V$.

</div>

Lineární zobrazení $f \colon U \to V$ je prosté, pokud $f(x) = f(y)$ nastane jenom pro $x = y$. Jinými slovy, pro každé dva vektory $x, y \in U$, $x \neq y$, platí $f(x) \neq f(y)$. Následující věta charakterizuje, kdy je lineární zobrazení prosté.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Věta 6.12 — Prosté lineární zobrazení)</span></p>

Buď $f \colon U \to V$ lineární zobrazení. Pak následující jsou ekvivalentní:

1. $f$ je prosté,
2. $\operatorname{Ker}(f) = \lbrace o \rbrace$,
3. obraz libovolné lineárně nezávislé množiny je lineárně nezávislá množina.

</div>

*Důkaz.* Dokážeme implikace $(1) \Rightarrow (2) \Rightarrow (3) \Rightarrow (1)$.

- Implikace $(1) \Rightarrow (2)$: Protože $f(o) = o$, tak $o \in \operatorname{Ker}(f)$. Ale vzhledem k tomu, že $f$ je prosté zobrazení, tak jádro už jiný prvek neobsahuje.
- Implikace $(2) \Rightarrow (3)$: Buďte $x_1, \ldots, x_n \in U$ lineárně nezávislé a nechť $\sum_{i=1}^{n} \alpha_i f(x_i) = o$. Pak $f(\sum_{i=1}^{n} \alpha_i x_i) = o$, čili $\sum_{i=1}^{n} \alpha_i x_i$ náleží do jádra $\operatorname{Ker}(f) = \lbrace o \rbrace$. Tudíž musí $\sum_{i=1}^{n} \alpha_i x_i = o$ a z lineární nezávislosti vektorů máme $\alpha_i = 0$ pro všechna $i$.
- Implikace $(3) \Rightarrow (1)$: Sporem předpokládejme, že existují dva různé vektory $x, y \in U$ takové, že $f(x) = f(y)$. Potom $o = f(x) - f(y) = f(x - y)$. Vektor $o$ představuje lineárně závislou množinu vektorů, tedy $x - y$ musí být podle předpokladu (3) také lineárně závislá množina, a tudíž $x - y = o$, neboli $x = y$. To je spor.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 6.13 — Prosté lineární zobrazení)</span></p>

Uvažujme lineární zobrazení z příkladu 6.9, tedy $x \mapsto Ax$, kde $A \in \mathbb{R}^{2 \times 2}$.

- Pro matici $A = \binom{-1 \ 0}{\ 0 \ 1}$ představuje zobrazení překlopení podle osy $x_2$. Protože jádro je $\operatorname{Ker}(f) = \lbrace o \rbrace$, zobrazení je prosté.
- Pro matici $A = \binom{1 \ 0}{0 \ 0}$ představuje zobrazení projekci na osu $x_1$. Protože jádro je $\operatorname{Ker}(f) = \operatorname{span}\lbrace (0, 1)^T \rbrace$, nejedná se o prosté zobrazení.

</div>

Speciálně, bod (3) věty 6.12 říká, že prosté lineární zobrazení $f \colon U \to V$ zobrazuje bázi prostoru $U$ na bázi $f(U)$. Tím pádem prosté zobrazení splňuje $\dim U = \dim f(U)$. Později (důsledek 6.43) uvidíme, že tato rovnost plně charakterizuje prostá zobrazení.

Ani prosté lineární zobrazení nemusí být vždy „na", o čemž svědčí kupříkladu zobrazení vnoření $\mathbb{R}^n$ do $\mathbb{R}^{n+1}$ definované předpisem $(v_1, \ldots, v_n)^T \mapsto (v_1, \ldots, v_n, 0)^T$.

U vektorových prostorů víme, že je každý (konečně generovaný) podprostor jednoznačně určený nějakou bází. Takovouto minimální reprezentaci bychom chtěli i pro lineární zobrazení. Jak uvidíme, jistá analogie platí i u lineárních zobrazení, protože každé lineární zobrazení je jednoznačně určeno tím, kam se zobrazí vektory z báze.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 6.14)</span></p>

Uvažujme lineární zobrazení $f \colon \mathbb{R}^2 \to V$. Pokud známe pouze obraz vektoru $x \neq o$, pak můžeme určit obrazy všech jeho násobků, tj. vektorů na přímce $\operatorname{span}\lbrace x \rbrace$, jednoduše ze vztahu $f(\alpha x) = \alpha f(x)$. Nedokážeme však zrekonstruovat celé zobrazení. K tomu potřebujeme znát ještě obraz nějakého jiného (lineárně nezávislého) vektoru $y$. Potom umíme dopočítat obraz nejen všech násobků vektorů $x$ a $y$, ale i jejich součtů a všech lineárních kombinací, tedy všech vektorů prostoru $\mathbb{R}^2$ ze vztahu $f(\alpha x + \beta y) = \alpha f(x) + \beta f(y)$. Tudíž lineární zobrazení $f$ je charakterizováno pouze obrazy dvou lineárně nezávislých vektorů, tedy báze.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Věta 6.15 — Lineární zobrazení a jednoznačnost vzhledem k obrazům báze)</span></p>

Buďte $U, V$ prostory nad $\mathbb{T}$ a $x_1, \ldots, x_n$ báze $U$. Pro libovolné vektory $y_1, \ldots, y_n \in V$ existuje právě jedno lineární zobrazení takové, že $f(x_i) = y_i$, $i = 1, \ldots, n$.

</div>

*Důkaz.* „Existence." Buď $x \in U$ libovolné. Pak $x = \sum_{i=1}^{n} \alpha_i x_i$ pro nějaké skaláry $\alpha_1, \ldots, \alpha_n \in \mathbb{T}$. Definujme obraz $x$ jako $f(x) = \sum_{i=1}^{n} \alpha_i y_i$, protože lineární zobrazení musí splňovat $f(x) = f\!\left(\sum_{i=1}^{n} \alpha_i x_i\right) = \sum_{i=1}^{n} \alpha_i f(x_i) = \sum_{i=1}^{n} \alpha_i y_i$. To, že takto definované zobrazení je lineární, se ověří už snadno.

„Jednoznačnost." Mějme dvě různá lineární zobrazení $f$ a $g$ splňující $f(x_i) = g(x_i) = y_i$ pro všechna $i = 1, \ldots, n$. Pak pro libovolné $x \in U$, které vyjádříme ve tvaru $x = \sum_{i=1}^{n} \alpha_i x_i$, je $f(x) = \sum_{i=1}^{n} \alpha_i f(x_i) = \sum_{i=1}^{n} \alpha_i y_i = \sum_{i=1}^{n} \alpha_i g(x_i) = g(x)$. Tedy $f(x) = g(x)$ $\forall x \in U$, což je spor s tím, že to jsou různá zobrazení.

### 6.2 Maticová reprezentace lineárního zobrazení

Každé lineární zobrazení mezi (konečně generovanými) vektorovými prostory jde reprezentovat maticově. Protože vektory mohou být rozličné objekty, je výhodné je popisovat v řeči souřadnic. Potom s nimi můžeme operovat jako s aritmetickými vektory, což je často pohodlnější.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 6.16 — Úvod k matici lineárního zobrazení)</span></p>

Uvažujme lineární zobrazení $f \colon \mathbb{T}^n \to \mathbb{T}^m$. Potom pro libovolné $x \in \mathbb{T}^n$ platí

$$f(x) = f\!\left(\sum_{i=1}^{n} x_i e_i\right) = \sum_{i=1}^{n} x_i f(e_i).$$

Označíme-li matici se sloupci $f(e_1), \ldots, f(e_n)$ jako

$$A = \begin{pmatrix} | & & | \\ f(e_1) & \cdots & f(e_n) \\ | & & | \end{pmatrix},$$

pak zřejmě $f(x) = Ax$. Každé lineární zobrazení $f \colon \mathbb{T}^n \to \mathbb{T}^m$ lze tedy reprezentovat maticově jako $f(x) = Ax$.

</div>

Uvažujme nyní lineární zobrazení $f \colon U \to \mathbb{T}^m$ a bázi $B = \lbrace v_1, \ldots, v_n \rbrace$ prostoru $U$. Nechť vektor $x \in U$ má vyjádření $x = \sum_{i=1}^{n} \alpha_i v_i$, tedy $[x]_B = (\alpha_1, \ldots, \alpha_n)^T$. Potom

$$f(x) = f\!\left(\sum_{i=1}^{n} \alpha_i v_i\right) = \sum_{i=1}^{n} \alpha_i f(v_i).$$

Označíme-li matici se sloupci $f(v_1), \ldots, f(v_n)$ jako

$$A = \begin{pmatrix} | & & | \\ f(v_1) & \cdots & f(v_n) \\ | & & | \end{pmatrix},$$

pak zřejmě $f(x) = A \cdot [x]_B$. Narozdíl od předchozího případu násobíme matici vektorem souřadnic $[x]_B$ vektoru $x$, a ne vektorem samotným.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Definice 6.17 — Matice lineárního zobrazení)</span></p>

Buď $f \colon U \to V$ lineární zobrazení, $B_U = \lbrace x_1, \ldots, x_n \rbrace$ báze prostoru $U$ nad $\mathbb{T}$ a $B_V = \lbrace y_1, \ldots, y_m \rbrace$ báze prostoru $V$ nad $\mathbb{T}$. Nechť $f(x_j) = \sum_{i=1}^{m} a_{ij} y_i$. Potom matice $A \in \mathbb{T}^{m \times n}$ s prvky $a_{ij}$, $i = 1, \ldots, m$, $j = 1, \ldots, n$, se nazývá *matice lineárního zobrazení* $f$ vzhledem k bázím $B_U, B_V$ a značí se ${}_{B_V}[f]_{B_U}$.

</div>

Jinými slovy, matice lineárního zobrazení vypadá tak, že její $j$-tý sloupec je tvořen souřadnicemi obrazu vektoru $x_j$ vzhledem k bázi $B_V$, to jest

$${}_{B_V}[f]_{B_U} = \begin{pmatrix} | & & | \\ [f(x_1)]_{B_V} & \cdots & [f(x_n)]_{B_V} \\ | & & | \end{pmatrix}.$$

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 6.18 — Matice lineárního zobrazení)</span></p>

Uvažujme lineární zobrazení $f \colon \mathbb{R}^2 \to \mathbb{R}^2$ s předpisem $f(x) = Ax$, kde

$$A = \begin{pmatrix} 1 & 2 \\ 3 & -4 \end{pmatrix}.$$

Zvolme báze $B_U = \lbrace (1, 2)^T, (2, 1)^T \rbrace$, $B_V = \lbrace (1, -1)^T, (0, 1)^T \rbrace$ a najděme matici zobrazení $f$ vzhledem k bázím $B_U, B_V$.

Obraz prvního vektoru báze $B_U$ je $f(1, 2) = (5, -5)^T$, a jeho souřadnice vzhledem k bázi $B_V$ jsou $[f(1, 2)]_{B_V} = (5, 0)^T$. Podobně, obraz druhého vektoru báze $B_U$ je $f(2, 1) = (4, 2)^T$, a jeho souřadnice vzhledem k bázi $B_V$ jsou $[f(2, 1)]_{B_V} = (4, 6)^T$. Tudíž

$${}_{B_V}[f]_{B_U} = \begin{pmatrix} 5 & 4 \\ 0 & 6 \end{pmatrix}.$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Věta 6.19 — Maticová reprezentace lineárního zobrazení)</span></p>

Buď $f \colon U \to V$ lineární zobrazení, $B_U = \lbrace x_1, \ldots, x_n \rbrace$ báze prostoru $U$, a $B_V = \lbrace y_1, \ldots, y_m \rbrace$ báze prostoru $V$. Pak pro každé $x \in U$ je

$$[f(x)]_{B_V} = {}_{B_V}[f]_{B_U} \cdot [x]_{B_U}. \tag{6.1}$$

</div>

*Důkaz.* Označme $A := {}_{B_V}[f]_{B_U}$. Buď $x \in U$, tedy $x = \sum_{i=1}^{n} \alpha_i x_i$, neboli $[x]_{B_U} = (\alpha_1, \ldots, \alpha_n)^T$. Pak

$$f(x) = f\!\left(\sum_{j=1}^{n} \alpha_j x_j\right) = \sum_{j=1}^{n} \alpha_j f(x_j) = \sum_{j=1}^{n} \alpha_j \left(\sum_{i=1}^{m} a_{ij} y_i\right) = \sum_{i=1}^{m} \left(\sum_{j=1}^{n} \alpha_j a_{ij}\right) y_i.$$

Tedy výraz $\sum_{j=1}^{n} \alpha_j a_{ij}$ reprezentuje $i$-tou souřadnici vektoru $[f(x)]_{B_V}$, ale jeho hodnota je $\sum_{j=1}^{n} \alpha_j a_{ij} = (A \cdot [x]_{B_U})_i$, což je $i$-tá složka vektoru ${}_{B_V}[f]_{B_U} \cdot [x]_{B_U}$.

Matice lineárního zobrazení tedy převádí souřadnice vektoru vzhledem k dané bázi na souřadnice jeho obrazu a navíc obraz libovolného vektoru můžeme vyjádřit jednoduchým způsobem jako násobení maticí.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(Důsledek 6.20)</span></p>

Každé lineární zobrazení $f \colon \mathbb{T}^n \to \mathbb{T}^m$ se dá vyjádřit jako $f(x) = Ax$ pro nějakou matici $A \in \mathbb{T}^{m \times n}$.

</div>

*Důkaz.* Pro každé $x \in \mathbb{T}^n$ je $f(x) = [f(x)]_{\text{kan}} = {}_{\text{kan}}[f]_{\text{kan}} \cdot [x]_{\text{kan}} = {}_{\text{kan}}[f]_{\text{kan}} \cdot x$. Tedy $f(x) = Ax$, kde $A = {}_{\text{kan}}[f]_{\text{kan}}$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Věta 6.21 — Jednoznačnost matice lineárního zobrazení)</span></p>

Buď $f \colon U \to V$ lineární zobrazení, $B_U$ báze prostoru $U$ a $B_V$ báze prostoru $V$. Pak jediná matice $A$ splňující (6.1) je $A = {}_{B_V}[f]_{B_U}$.

</div>

*Důkaz.* Nechť báze $B_U$ sestává z vektorů $z_1, \ldots, z_n$. Pro spor předpokládejme, že lineární zobrazení $f$ má dvě maticové reprezentace (6.1) pomocí matic $A \neq A'$. Tudíž existuje takový vektor $s \in \mathbb{T}^n$ takový, že $As \neq A's$; takový vektor lze volit například jako jednotkový vektor s jedničkou na takové pozici, ve které sloupci se matice $A, A'$ liší. Definujme vektor $x := \sum_{i=1}^{n} s_i z_i$. Pak $[f(x)]_{B_V} = As \neq A's = [f(x)]_{B_V}$, což je spor s jednoznačností souřadnic (věta 5.33).

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Poznámka 6.22)</span></p>

Nejenže každé lineární zobrazení jde reprezentovat maticově, ale i naopak každá matice představuje matici nějakého lineárního zobrazení. Buďte $B_U, B_V$ báze prostorů $U, V$ dimenzí $n, m$ a mějme $A \in \mathbb{T}^{m \times n}$. Pak existuje jediné lineární zobrazení $f \colon U \to V$ takové, že $A = {}_{B_V}[f]_{B_U}$; ve sloupcích matice $A$ vyčteme souřadnice obrazů vektorů báze $B_U$, což plně určuje zobrazení $f$ dle věty 6.15. To znamená, že existuje vzájemně jednoznačná korespondence mezi lineárními zobrazeními $f \colon U \to V$ a prostorem matic $\mathbb{T}^{m \times n}$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Definice 6.23 — Matice přechodu)</span></p>

Buď $V$ vektorový prostor a $B_1, B_2$ dvě jeho báze. Pak *maticí přechodu* od $B_1$ k $B_2$ nazveme matici ${}_{B_2}[id]_{B_1}$.

</div>

Matice přechodu má pak podle maticové reprezentace tento význam: Buď $x \in U$, pak

$$[x]_{B_2} = {}_{B_2}[id]_{B_1} \cdot [x]_{B_1},$$

tedy pouhým maticovým násobením získáváme souřadnice vzhledem k jiné bázi. Zřejmě platí ${}_{B}[id]_{B} = I_n$ pro libovolnou bázi $B$.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 6.24 — Matice přechodu)</span></p>

Najděte matici přechodu v $\mathbb{R}^3$ od báze

$$B_1 = \lbrace (1, 1, -1)^T, (3, -2, 0)^T, (2, -1, 1)^T \rbrace$$

k bázi

$$B_2 = \lbrace (8, -4, 1)^T, (-8, 5, -2)^T, (3, -2, 1)^T \rbrace.$$

Řešení: spočítáme

$$[(1, 1, -1)^T]_{B_2} = (2, 3, 3)^T, \quad [(3, -2, 0)^T]_{B_2} = (-1, -4, -7)^T, \quad [(2, -1, 1)^T]_{B_2} = (1, 3, 6)^T.$$

Tedy

$${}_{B_2}[id]_{B_1} = \begin{pmatrix} 2 & -1 & 1 \\ 3 & -4 & 3 \\ 3 & -7 & 6 \end{pmatrix}.$$

Víme-li například, že souřadnice vektoru $(4, -1, -1)^T$ vzhledem k bázi $B_1$ jsou $(1, 1, 0)^T$, pak souřadnice vzhledem k $B_2$ získáme

$$[(4, -1, -1)^T]_{B_2} = {}_{B_2}[id]_{B_1} \cdot [(4, -1, -1)^T]_{B_1} = {}_{B_2}[id]_{B_1} \cdot (1, 1, 0)^T = (1, -1, -4)^T.$$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 6.25)</span></p>

Buď $B$ báze prostoru $\mathbb{T}^n$. Podle maticové reprezentace lineárního zobrazení pak speciálně dostaneme

$$[x]_B = {}_{B}[id]_{\text{kan}} \cdot [x]_{\text{kan}} = {}_{B}[id]_{\text{kan}} \cdot x.$$

Souřadnice libovolného vektoru tudíž získáme jednoduše vynásobením matice přechodu s vektorem $x$.

</div>

Podstatnou roli v teorii lineárních zobrazení hraje jejich vzájemné skládání. Připomeňme, že pro zobrazení $f \colon U \to V$ a $g \colon V \to W$ je složené zobrazení $g \circ f$ definované předpisem $(g \circ f)(x) := g(f(x))$, $x \in U$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Tvrzení 6.26 — Složené lineární zobrazení)</span></p>

Buďte $f \colon U \to V$, $g \colon V \to W$ lineární zobrazení. Pak složené zobrazení $g \circ f$ je zase lineární zobrazení.

</div>

Uvažujme dvě lineární zobrazení $f \colon \mathbb{T}^n \to \mathbb{T}^p$ a $g \colon \mathbb{T}^p \to \mathbb{T}^m$ reprezentovaná maticově $f(x) = Ax$, $g(y) = By$ pro určité matice $A \in \mathbb{T}^{p \times n}$, $B \in \mathbb{T}^{m \times p}$. Potom složené zobrazení má předpis

$$(g \circ f)(x) = g(f(x)) = B(Ax) = (BA)x.$$

Je to tedy lineární zobrazení reprezentované maticí $BA$ (viz poznámka 3.20). Tato vlastnost platí obecněji, matice složeného lineárního zobrazení je rovna součinu matic příslušných zobrazení.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Věta 6.27 — Matice složeného lineárního zobrazení)</span></p>

Buďte $f \colon U \to V$ a $g \colon V \to W$ lineární zobrazení, buď $B_U$ báze $U$, $B_V$ báze $V$ a $B_W$ báze $W$. Pak

$${}_{B_W}[g \circ f]_{B_U} = {}_{B_W}[g]_{B_V} \cdot {}_{B_V}[f]_{B_U}. \tag{6.2}$$

</div>

*Důkaz.* Pro každé $x \in U$ je

$$[(g \circ f)(x)]_{B_W} = [g(f(x))]_{B_W} = {}_{B_W}[g]_{B_V} \cdot [f(x)]_{B_V} = {}_{B_W}[g]_{B_V} \cdot {}_{B_V}[f]_{B_U} \cdot [x]_{B_U}.$$

Díky jednoznačnosti matice lineárního zobrazení (věta 6.21) je ${}_{B_W}[g]_{B_V} \cdot {}_{B_V}[f]_{B_U}$ hledaná matice složeného zobrazení.

Ve vzorečku (6.2) se opět uplatní mnemotechnika ve značení matic lineárních zobrazení. Konkrétně, matice zobrazení $g \circ f$ má na vstupu stejnou bázi $B_U$ jako matice zobrazení $f$ a na výstupu stejnou bázi $B_W$ jako matice zobrazení $g$. Navíc výstupní báze $B_V$ matice zobrazení $f$ musí být stejná jako vstupní báze matice zobrazení $g$.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 6.28 — Skládání otočení a součtové vzorce pro sin a cos)</span></p>

Otočení v rovině o úhel $\alpha$ proti směru hodinových ručiček má vzhledem ke kanonické bázi matici

$$\begin{pmatrix} \cos\alpha & -\sin\alpha \\ \sin\alpha & \cos\alpha \end{pmatrix},$$

viz příklad 6.3. Podobně otočení o úhel $\beta$. Matici otočení o úhel $\alpha + \beta$ můžeme získat přímo dosazením hodnoty $\alpha + \beta$ do matice rotace nebo složením otočení o úhel $\alpha$ a pak otočení o úhel $\beta$. Porovnáním získáme součtové vzorce pro sin a cos:

$$\begin{pmatrix} \cos(\alpha + \beta) & -\sin(\alpha + \beta) \\ \sin(\alpha + \beta) & \cos(\alpha + \beta) \end{pmatrix} = \begin{pmatrix} \cos\beta & -\sin\beta \\ \sin\beta & \cos\beta \end{pmatrix} \begin{pmatrix} \cos\alpha & -\sin\alpha \\ \sin\alpha & \cos\alpha \end{pmatrix} = \begin{pmatrix} \cos\alpha\cos\beta - \sin\alpha\sin\beta & -\sin\alpha\cos\beta - \sin\beta\cos\alpha \\ \cos\alpha\sin\beta + \cos\beta\sin\alpha & -\sin\alpha\sin\beta + \cos\alpha\cos\beta \end{pmatrix}.$$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 6.29 — Přepočet matice zobrazení mezi bázemi)</span></p>

Nechť máme dánu matici lineárního zobrazení $f$ vzhledem k bázím $B_1, B_2$, tj. ${}_{B_2}[f]_{B_1}$. Jak určit matici vzhledem k bázím $B_3, B_4$, tj. ${}_{B_4}[f]_{B_3}$? Podle věty o matici složeného zobrazení aplikované na $f = id \circ f \circ id$ a příslušné báze máme

$${}_{B_4}[f]_{B_3} = {}_{B_4}[id]_{B_2} \cdot {}_{B_2}[f]_{B_1} \cdot {}_{B_1}[id]_{B_3}.$$

Tedy veškerou práci vykonají matice přechodu mezi bázemi.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Poznámka 6.30 — Derivace více proměnných a skládání zobrazení)</span></p>

Uvažujme diferencovatelné funkce $f(x) \colon \mathbb{R}^n \to \mathbb{R}^p$, $g(y) \colon \mathbb{R}^p \to \mathbb{R}^m$ a body $x^* \in \mathbb{R}^n$ a $y^* := f(x^*)$. Z kursu diferenciálního počtu více proměnných známe formuli pro parciální derivace složeného zobrazení

$$\frac{\partial (g \circ f)_i}{\partial x_k} = \sum_{j=1}^{p} \frac{\partial g_i}{\partial y_j} \cdot \frac{\partial f_j}{\partial x_k}.$$

V řeči Jacobiho matic

$$\nabla f = \begin{pmatrix} \frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_n} \\ \vdots & & \vdots \\ \frac{\partial f_p}{\partial x_1} & \cdots & \frac{\partial f_p}{\partial x_n} \end{pmatrix}$$

má výše zmíněná formule tvar

$$\nabla (g \circ f)(x^*) = \nabla g(y^*) \cdot \nabla f(x^*). \tag{6.3}$$

Jacobiho matice je matice lineárního zobrazení (lokálně nejlépe) aproximujícího hladké zobrazení. Formule (6.3) říká, že při skládání hladkých zobrazení se odpovídajícím způsobem skládají i jejich lineární aproximace. Formule tím také ilustruje větu 6.27 o matici složeného lineárního zobrazení.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 6.31 — Posunutí jako lineární zobrazení?)</span></p>

Buď $v \in \mathbb{R}^n$ pevné a uvažujme zobrazení $f \colon \mathbb{R}^n \to \mathbb{R}^n$ dané předpisem $f(x) = x + v$. Toto zobrazení není lineární, protože nezobrazuje nulový vektor na nulový vektor. Nicméně můžeme ho jako lineární simulovat využitím technik z klasické projektivní geometrie. Vnoříme prostor $\mathbb{R}^n$ do prostoru o jednu dimenzi většího tak, aby pro určité lineární zobrazení $g \colon \mathbb{R}^{n+1} \to \mathbb{R}^{n+1}$ platilo

$$g(x_1, \ldots, x_n, 1) = (x_1 + v_1, \ldots, x_n + v_n, 1).$$

Dodefinujeme $g$ pro ostatní body tak, aby tvořilo lineární zobrazení

$$g(x_1, \ldots, x_n, x_{n+1}) = (x_1 + v_1 x_{n+1}, \ldots, x_n + v_n x_{n+1}, x_{n+1}).$$

Matice tohoto zobrazení je

$$\begin{pmatrix} 1 & 0 & \cdots & 0 & v_1 \\ 0 & 1 & \cdots & 0 & v_2 \\ \vdots & & \ddots & & \vdots \\ 0 & 0 & \cdots & 1 & v_n \\ 0 & 0 & \cdots & 0 & 1 \end{pmatrix}.$$

</div>

### 6.3 Isomorfismus

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Definice 6.32 — Isomorfismus)</span></p>

*Isomorfismus* mezi prostory $U, V$ nad tělesem $\mathbb{T}$ je vzájemně jednoznačné lineární zobrazení $f \colon U \to V$. Pokud mezi prostory $U, V$ existuje isomorfismus, pak říkáme, že $U, V$ jsou *isomorfní*.

</div>

Isomorfní prostory se chovají z pohledu lineární algebry stejně. Isomorfismus zobrazuje lineárně závislé vektory na lineárně závislé se stejnými vztahy (protože jde o lineární zobrazení), zobrazuje lineárně nezávislé na lineárně nezávislé (protože jde o prosté zobrazení), zachovává dimenzi, zobrazuje bázi na bázi atp.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 6.33)</span></p>

Příkladem isomorfismu je třeba škálování, překlopení v $\mathbb{R}^2$ (příklad 6.2) nebo otáčení (příklad 6.28). Příkladem lineárního zobrazení, které není isomorfismem, je projekce (příklad 6.2).

Příkladem isomorfních prostorů je například $\mathcal{P}^n$ a $\mathbb{R}^{n+1}$, kdy vhodným (a nikoliv jediným) isomorfismem je

$$a_n x^n + \ldots + a_1 x + a_0 \mapsto (a_n, \ldots, a_1, a_0).$$

Jiným příkladem isomorfních prostorů je $\mathbb{R}^{m \times n}$ a $\mathbb{R}^{mn}$, kdy vhodným isomorfismem je například

$$A \mapsto (a_{11}, \ldots, a_{1n}, a_{21}, \ldots, a_{2n}, \ldots, a_{m1}, \ldots, a_{mn}).$$

Vektorový prostor $\mathbb{C}^n$ nad $\mathbb{R}$ je isomorfní prostoru $\mathbb{R}^{2n}$ nad $\mathbb{R}$. Konkrétní isomorfismus je například zobrazení, které vektor $(a_1 + ib_1, a_n + ib_n)^T \in \mathbb{C}^n$ zobrazuje na vektor $(a_1, b_1, \ldots, a_n, b_n)^T \in \mathbb{R}^{2n}$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Věta 6.34 — Vlastnosti isomorfismu)</span></p>

1. Je-li $f \colon U \to V$ isomorfismus, pak $f^{-1} \colon V \to U$ existuje a je to také isomorfismus.
2. Jsou-li $f \colon U \to V$ a $g \colon V \to W$ isomorfismy, pak $g \circ f \colon U \to W$ je také isomorfismus.
3. Lineární zobrazení $f \colon U \to V$ je isomorfismem právě tehdy, když libovolná báze prostoru $U$ se zobrazuje na bázi prostoru $V$.
4. Je-li $f \colon U \to V$ isomorfismus, pak $\dim U = \dim V$.

</div>

*Důkaz.*

1. Zobrazení $f$ je vzájemně jednoznačné, tedy $f^{-1}$ existuje a je také vzájemně jednoznačné. Zbývá dokázat linearitu. Buď $v_1, v_2 \in V$ a nechť $f^{-1}(v_1) = u_1$ a $f^{-1}(v_2) = u_2$. Pak $f(u_1 + u_2) = f(u_1) + f(u_2) = v_1 + v_2$, tedy $f^{-1}(v_1 + v_2) = u_1 + u_2 = f^{-1}(v_1) + f^{-1}(v_2)$. Podobně pro násobky: Nechť $v \in V$ a $f^{-1}(v) = u$, pak $f(\alpha u) = \alpha f(u) = \alpha v$, tedy $f^{-1}(\alpha v) = \alpha u = \alpha f^{-1}(v)$.
2. Snadné z tvrzení 6.26.
3. Buď $x_1, \ldots, x_n$ báze $U$. Protože $f$ je prosté, dle věty 6.12(3) jsou obrazy $f(x_1), \ldots, f(x_n)$ lineárně nezávislé. Protože $f$ je na, generují vektory $f(x_1), \ldots, f(x_n)$ dle tvrzení 6.10(3) prostor $f(U) = V$. Tedy vektory $f(x_1), \ldots, f(x_n)$ tvoří bázi $V$. — Naopak, buď $x_1, \ldots, x_n$ báze $U$ a $f(x_1), \ldots, f(x_n)$ báze $V$. Pak zobrazení $f$ je zřejmě na. To, že zobrazení $f$ je prosté, nahlédneme sporem: Předpokládejme, že jádro $\operatorname{Ker}(f)$ obsahuje nenulový vektor. Tudíž pro nějakou netriviální lineární kombinaci platí $f(\sum_{i=1}^{n} \alpha_i x_i) = o$. Z linearity zobrazení dostáváme $\sum_{i=1}^{n} \alpha_i f(x_i) = o$, což je spor s lineární nezávislostí vektorů $f(x_1), \ldots, f(x_n)$.
4. Plyne z předchozího bodu.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Tvrzení 6.35)</span></p>

Buď $f \colon U \to V$ isomorfismus, $B_U$ báze $U$ a $B_V$ báze $V$. Pak

$${}_{B_U}[f^{-1}]_{B_V} = {}_{B_V}[f]_{B_U}^{-1}.$$

</div>

*Důkaz.* Protože $f^{-1} \circ f = id$, dostáváme

$${}_{B_U}[f^{-1}]_{B_V} \cdot {}_{B_V}[f]_{B_U} = {}_{B_U}[f^{-1} \circ f]_{B_U} = {}_{B_U}[id]_{B_U} = I.$$

Jelikož ${}_{B_V}[f]_{B_U}$ je podle věty 6.34(4) čtvercová, je ${}_{B_U}[f^{-1}]_{B_V}$ její inverzní matice.

Matice isomorfismu má matici inverzní, tedy musí být regulární. Toto tvrzení platí i naopak: Je-li matice lineárního zobrazení $f$ regulární, pak je $f$ isomorfismem, protože inverzní matice dává předpis pro inverzní zobrazení $f^{-1}$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Tvrzení 6.36)</span></p>

Lineární zobrazení $f \colon U \to V$ je isomorfismus právě tehdy, když nějaká (libovolná) matice reprezentující $f$ je regulární.

</div>

Další důsledek tvrzení 6.35 dostaneme speciálně pro matici přechodu mezi bázemi $B_U$ a $B_V$, a to

$${}_{B_U}[id]_{B_V} = {}_{B_V}[id]_{B_U}^{-1}.$$

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 6.37 — Mnemotechnika počítání matic přechodu)</span></p>

Pro počítání matice přechodu v $\mathbb{R}^n$ od báze $B_U$ do báze $B_V$, tj. ${}_{B_V}[id]_{B_U}$, lze použít následující mnemotechniku:

$$(B_V \mid B_U) \xrightarrow{\text{RREF}} (I_n \mid {}_{B_V}[id]_{B_U}).$$

První matice ve sloupcích obsahuje bázi $B_V$ a pak bázi $B_U$, což jsou vlastně matice ${}_{\text{kan}}[id]_{B_V}$ a ${}_{\text{kan}}[id]_{B_U}$. Převedením na RREF tvar dostaneme napravo hledanou matici přechodu. Důvod pramení ze vztahu ${}_{B_V}[id]_{B_U} = {}_{B_V}[id]_{\text{kan}} \cdot {}_{\text{kan}}[id]_{B_U} = {}_{\text{kan}}[id]_{B_V}^{-1} \cdot {}_{\text{kan}}[id]_{B_U}$. Převedení matice na RREF tvar lze vyjádřit vynásobením maticí ${}_{\text{kan}}[id]_{B_V}^{-1}$ zleva.

Konkrétně, pro příklad 6.24, dostaneme

$$\begin{pmatrix} 8 & -8 & 3 & 1 & 3 & 2 \\ -4 & 5 & -2 & 1 & -2 & -1 \\ 1 & -2 & 1 & -1 & 0 & 1 \end{pmatrix} \xrightarrow{\text{RREF}} \begin{pmatrix} 1 & 0 & 0 & 2 & -1 & 1 \\ 0 & 1 & 0 & 3 & -4 & 3 \\ 0 & 0 & 1 & 3 & -7 & 6 \end{pmatrix}.$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Tvrzení 6.38)</span></p>

Buď $V$ vektorový prostor nad tělesem $\mathbb{T}$ dimenze $n$ s bází $B$. Pak zobrazení $x \mapsto [x]_B$ je isomorfismus mezi prostory $V$ a $\mathbb{T}^n$.

</div>

*Důkaz.* Nechť báze $B$ sestává z vektorů $v_1, \ldots, v_n$. Snadno se nahlédne, že zobrazení $x \mapsto [x]_B$ je lineární, že je prosté a že je „na", protože každá $n$-tice $(\alpha_1, \ldots, \alpha_n) \in \mathbb{T}^n$ představuje souřadnice, konkrétně vektoru $\sum_{i=1}^{n} \alpha_i v_i$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Věta 6.39 — Isomorfismus $n$-dimenzionálních prostorů)</span></p>

Všechny $n$-dimenzionální vektorové prostory nad tělesem $\mathbb{T}$ jsou navzájem isomorfní.

</div>

*Důkaz.* Podle tvrzení 6.38 jsou všechny $n$-dimenzionální vektorové prostory nad tělesem $\mathbb{T}$ isomorfní s $\mathbb{T}^n$, a tím pádem i navzájem mezi sebou, neboť složení isomorfismů je zase isomorfismus.

Věta říká, že všechny $n$-dimenzionální prostory nad stejným tělesem jsou navzájem isomorfní. To znamená, že jsou z určitého pohledu stejné. Přestože každý má svá specifika, zvláštní operace atp., vykazují podobnou strukturu a můžeme k nim přistupovat jednotným způsobem. Tudíž při hledání dimenze, ověřování lineární nezávislosti atp. stačí přejít isomorfismem do prostoru $\mathbb{T}^n$ nad $\mathbb{T}$, kde se pracuje mnohem lépe. Isomorfismus totiž zachovává lineární nezávislost vektorů, zachovává dimenzi obrazu podprostoru, a také zachovává závislost vektorů.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 6.40)</span></p>

Uvažujme polynomy $2x^3 + x^2 + x + 3$, $x^3 + 2x^2 + 3x + 1$, $x^3 - x^2 - 2x + 2$, $4x^3 - x^2 - 3x + 7$ jako vektory prostoru $\mathcal{P}^3$. Jsou lineárně nezávislé? Jakou dimenzi má prostor jimi generovaný? Jaká je jeho báze? Na tyto otázky snadno odpovíme při použití isomorfismu $a_3 x^3 + a_2 x^2 + a_1 x + a_0 \mapsto (a_3, a_2, a_1, a_0)$. Takto se polynomy zobrazí na vektory

$$(2, 1, 1, 3)^T, \quad (1, 2, 3, 1)^T, \quad (1, -1, -2, 2)^T, \quad (4, -1, -3, 7)^T.$$

Nyní již standardním způsobem (příklad 5.70) zjistíme, že vektory (a tedy i polynomy) jsou lineárně závislé, generují dvoudimenzionální podprostor a bázi tvoří například první dva.

</div>

Pro lineární zobrazení $f \colon \mathbb{R}^n \to \mathbb{R}^m$ definované předpisem $f(x) = Ax$ platí $\operatorname{Ker}(f) = \operatorname{Ker}(A)$ a $f(\mathbb{R}^n) = \mathcal{S}(A)$. I v obecném případě je úzký vztah mezi jádrem lineárního zobrazení a jádrem příslušné matice a podobně mezi obrazem a sloupcovým prostorem matice.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Věta 6.41 — O dimenzi jádra a obrazu)</span></p>

Buď $f \colon U \to V$ lineární zobrazení, $B_U$ báze prostoru $U$ a $B_V$ báze prostoru $V$. Označme $A = {}_{B_V}[f]_{B_U}$. Pak:

1. $\dim \operatorname{Ker}(f) = \dim \operatorname{Ker}(A)$,
2. $\dim f(U) = \dim \mathcal{S}(A) = \operatorname{rank}(A)$.

</div>

*Důkaz.*

1. Podle věty 6.34(4) stačí sestrojit isomorfismus mezi prostory $\operatorname{Ker}(f)$ a $\operatorname{Ker}(A)$. Isomorfismem může být např. zobrazení $x \in \operatorname{Ker}(f) \mapsto [x]_{B_U}$. Z tvrzení 6.38 víme, že je lineární a prosté. Zbývá ukázat, že $[x]_{B_U} \in \operatorname{Ker}(A)$ a že zobrazení je „na". Buď $x \in \operatorname{Ker}(f)$, pak $o = [f(x)]_{B_V} = {}_{B_V}[f]_{B_U} \cdot [x]_{B_U}$, tedy $[x]_{B_U} \in \operatorname{Ker}(A)$. Také naopak, pro každé $[x]_{B_U} \in \operatorname{Ker}(A)$ je $f(x) = o$.
2. Označme $\dim U = n$, $\dim V = m$. Opět sestrojíme isomorfismus, nyní mezi $f(U)$ a $\mathcal{S}(A)$, a to takto $y \in f(U) \mapsto [y]_{B_V}$. A opět, zobrazení je lineární a prosté. Dále, pro $y \in f(U)$ existuje $x \in U$ takové, že $f(x) = y$. Nyní $[y]_{B_V} = [f(x)]_{B_V} = A \cdot [x]_{B_U}$, tedy $[y]_{B_V}$ náleží do sloupcového prostoru $\mathcal{S}(A)$. A naopak, pro každé $b \in \mathcal{S}(A)$ existuje $a \in \mathbb{T}^n$ takové, že $b = Aa$. Čili pro vektor $x \in U$ takový, že $[x]_{B_U} = a$, platí $y := f(x) \in f(U)$ a zároveň $[y]_{B_V} = [f(x)]_{B_V} = A \cdot [x]_{B_U} = Aa = b \in \mathcal{S}(A)$.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Poznámka 6.42)</span></p>

Důkaz věty 6.41 je konstruktivní — říká nejen jak spočítat dimenzi jádra a obrazu $f$, ale také jak najít jejich báze. Je-li $x_1, \ldots, x_k$ báze $\operatorname{Ker}(A)$, pak tyto vektory tvoří souřadnice (vzhledem k bázi $B_U$) báze $\operatorname{Ker}(f)$. Podobně, je-li $y_1, \ldots, y_r$ báze prostoru $\mathcal{S}(A)$, pak tyto vektory představují souřadnice báze prostoru $f(U)$ vzhledem k $B_V$.

</div>

Jako důsledek věty 6.41 dostáváme následující zobecnění rovnosti z věty 6.34(4), neboť pro isomorfismus máme $\dim \operatorname{Ker}(f) = 0$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(Důsledek 6.43)</span></p>

Buď $f \colon U \to V$ lineární zobrazení, pak $\dim U = \dim \operatorname{Ker}(f) + \dim f(U)$.

</div>

*Důkaz.* Podle věty 5.72 platí pro matici $A$ typu $m \times n$ rovnost $n = \dim \operatorname{Ker}(A) + \operatorname{rank}(A)$. Speciálně, pro $A = {}_{B_V}[f]_{B_U}$ dostáváme hledanou identitu, neboť $n = \dim U$, $\dim \operatorname{Ker}(f) = \dim \operatorname{Ker}(A)$ a $\dim f(U) = \operatorname{rank}(A)$.

Již na straně 102 jsme nahlédli, že jádro lineárního zobrazení popisuje jak moc zobrazení degeneruje. Důsledek 6.43 pak vyjadřuje míru degenerace číselně. Dimenze jádra udává rozdíl mezi dimenzí prostoru $U$ a dimenzí jeho obrazu.

S ohledem na věty 6.12 a 6.41 pak dostáváme, že lineární zobrazení $f \colon U \to V$ je prosté právě tehdy, když $\dim U = \dim f(U)$, neboli $\dim U = \operatorname{rank}({}_{B_V}[f]_{B_U})$. Nutná a postačující podmínka pro to, aby $f$ bylo prosté, tedy je, aby matice zobrazení $f$ vzhledem k libovolným bázím měla lineárně nezávislé sloupce.

Jak poznáme, že lineární zobrazení $f \colon U \to V$ je „na"? Tuto situaci můžeme vyjádřit podmínkou $\dim V = \dim f(U)$, neboli $\dim V = \operatorname{rank}({}_{B_V}[f]_{B_U})$. Ekvivalentně tedy matice $f$ vzhledem k libovolným bázím musí mít lineárně nezávislé řádky. Dostáváme tedy následující tvrzení.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Tvrzení 6.44)</span></p>

Buď $f \colon U \to V$ lineární zobrazení, $B_U$ báze prostoru $U$ a $B_V$ báze prostoru $V$. Pak:

1. $f$ je prosté právě tehdy, když ${}_{B_V}[f]_{B_U}$ má lineárně nezávislé sloupce,
2. $f$ je „na" právě tehdy, když ${}_{B_V}[f]_{B_U}$ má lineárně nezávislé řádky.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 6.45)</span></p>

Mějme lineární zobrazení $f \colon \mathbb{R}^3 \to \mathcal{P}^2$ dané maticí

$${}_{B_V}[f]_{B_U} = A = \begin{pmatrix} 1 & 1 & 1 \\ 3 & 2 & 0 \\ 0 & 1 & 3 \end{pmatrix},$$

kde

$$B_U = \lbrace (1, 2, 1)^T, \ (0, 1, 1)^T, \ (1, 2, 4)^T \rbrace, \qquad B_V = \lbrace x^2 - 2x + 3, \ x - 1, \ 2x^2 + x \rbrace.$$

Protože $\operatorname{rank}(A) = 2$, dostáváme ihned, že $\dim \operatorname{Ker}(f) = 3 - \operatorname{rank}(A) = 1$ a $\dim f(\mathbb{R}^3) = \operatorname{rank}(A) = 2$. Jelikož má jádro kladnou dimenzi a je tudíž netriviální, podle věty 6.12 to znamená, že zobrazení $f$ není prosté. Jelikož má obraz dimenzi 2, ale prostor $\mathcal{P}^2$ má dimenzi 3, tak zobrazení $f$ není „na".

Báze $\operatorname{Ker}(A)$ je $(2, -3, 1)^T$, což reprezentuje souřadnice hledaného vektoru v bázi $B_U$. Tedy báze $\operatorname{Ker}(f)$ je tvořena vektorem $2(1, 2, 1)^T - 3(0, 1, 1)^T + 1(1, 2, 4)^T = (3, 3, 3)^T$.

Báze $\mathcal{S}(A)$ je $(1, 3, 0)^T$, $(1, 2, 1)^T$, což opět reprezentuje souřadnice hledaných vektorů. Tudíž báze obrazu $f(\mathbb{R}^3)$ tvoří dva vektory

$$1(x^2 - 2x + 3) + 3(x - 1) + 0(2x^2 + x) = x^2 + x, \qquad 1(x^2 - 2x + 3) + 2(x - 1) + 1(2x^2 + x) = 3x^2 + x + 1.$$

</div>

### 6.4 Prostor lineárních zobrazení

Není těžké nahlédnout, že množina lineárních zobrazení z prostoru $U$ nad $\mathbb{T}$ dimenze $n$ do prostoru $V$ nad $\mathbb{T}$ dimenze $m$ tvoří vektorový prostor: součet lineárních zobrazení $f, g \colon U \to V$ je opět lineární zobrazení $(f + g) \colon U \to V$ a násobek $\alpha f$ lineárního zobrazení $f \colon U \to V$ je také lineární zobrazení. Nulovým vektorem je zobrazení $u \mapsto o_V$ $\forall u \in U$.

Navíc, protože každé lineární zobrazení je jednoznačně určeno maticí vzhledem k daným bázím, je tento prostor lineárních zobrazení isomorfní s prostorem matic $\mathbb{T}^{m \times n}$ a má tedy dimenzi $mn$. Příslušným isomorfismem pak může být zobrazení $f \mapsto {}_{B_V}[f]_{B_U}$, kde $B_U$ je libovolná pevná báze prostoru $U$ a $B_V$ je libovolná pevná báze prostoru $V$. Linearita tohoto zobrazení plyne jednoduše (díky linearitě souřadnic) z vlastností

$${}_{B_V}[f + g]_{B_U} = {}_{B_V}[f]_{B_U} + {}_{B_V}[g]_{B_U}, \qquad {}_{B_V}[\alpha f]_{B_U} = \alpha \, {}_{B_V}[f]_{B_U}.$$

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Definice 6.46 — Lineární forma a duální prostor)</span></p>

Buď $V$ vektorový prostor nad $\mathbb{T}$. Pak *lineární forma* (nebo též lineární funkcionál) je libovolné lineární zobrazení z $V$ do $\mathbb{T}$. *Duální prostor*, značený $V^*$, je vektorový prostor všech lineárních forem.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 6.47)</span></p>

Lineární formou na prostoru $\mathbb{R}^n$ nad $\mathbb{R}$ je například zobrazení $f(x_1, \ldots, x_n) = \frac{1}{n}\sum_{i=1}^{n} x_i$ nebo zobrazení $g(x_1, \ldots, x_n) = x_1$.

</div>

Nic nám nebrání uvažovat duální prostor k duálnímu prostoru. Je to tedy prostor $V^{**}$ všech lineárních zobrazení $F \colon V^* \to \mathbb{T}$. Jinými slovy, $F$ každou lineární formu na $V$ zobrazí na skalár z tělesa $\mathbb{T}$. Například, buď $v^* \in V$ pevný vektor a uvažujme zobrazení, které lineární formu $f$ zobrazí na její funkční hodnotu $f(v^*)$. Právě jsme definovali zobrazení $F_{v^*} \in V^{**}$ dané předpisem $F_{v^*}(f) = f(v^*)$. Ke každému vektoru $v^* \in V$ jsme takto našli vektor $F_{v^*} \in V^{**}$. Zobrazení $v^* \mapsto F_{v^*}$ se nazývá *kanonické vnoření* prostoru $V$ do prostoru $V^{**}$. Dá se ukázat, že je to prosté lineární zobrazení.

Je-li $\dim V = n$, pak také $\dim V^* = n$. Je-li $v_1, \ldots, v_n$ báze $V$, pak duální prostor má například bázi $f_1, \ldots, f_n$, kde $f_i$ je určeno obrazy báze $f_i(v_i) = 1$ a $f_i(v_j) = 0$ pro $i \neq j$. Tato báze se nazývá *duální báze* k bázi $v_1, \ldots, v_n$.

Pro konečně generovaný prostor je tedy $V$ isomorfní s duálním prostorem $V^*$, s duálem k duálnímu prostoru $V^{**}$ atd. Nicméně vždy existuje kanonické vnoření $V$ do $V^{**}$. Pokud navíc platí, že $V$ a $V^{**}$ jsou isomorfní, tak navíc platí, že $V$ má určité pěkné vlastnosti.

### 6.5 Aplikace

Lineární zobrazení mají široké uplatnění v počítačové grafice pro vizualizaci dat, animaci, modelování 3D scén atp. Protože lineární zobrazení umožňuje pomocí jednoduchých maticových operací provádět základní transformace (škálování, otáčení, projekce, …), dostáváme tím elegantní způsob jak zobrazovat dvou a trojdimenzionální objekty.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 6.48 — Vizualizace trojrozměrných objektů)</span></p>

Uvažujme objekt pro vizualizaci; v praxi to může být například 3D obraz lidského orgánu pomocí magnetické rezonance nebo CT, který chceme z určitého pohledu a v určitém měřítku zobrazit. Objekt je umístěný v zadaném souřadném systému. V našem případě uvažujeme objekt ve tvaru válce se středem podstavy v počátku.

Nejprve je nutné objekt přeškálovat, aby měl požadovanou velikost. To provedeme transformací $x \mapsto Ax$ s diagonální maticí $A = \operatorname{diag}(\alpha_x, \alpha_y, \alpha_z)$. V ose $x$ škálujeme s koeficientem $\alpha_x$ a podobně pro ostatní osy. Při rovnoměrném škálování je $A = \alpha I_3$.

Dále je potřeba objekt umístit na správné místo a natočit ho do správné pozice. Každé otočení v prostoru $\mathbb{R}^3$ lze složit ze tří rotací kolem souřadných os. Podle příkladu 6.3 má matice rotace kolem osy $y$ o úhel $\varphi$ tvar

$$\begin{pmatrix} \cos(\varphi) & 0 & -\sin(\varphi) \\ 0 & 1 & 0 \\ \sin(\varphi) & 0 & \cos(\varphi) \end{pmatrix}.$$

Nakonec provedeme projekci objektu na příslušnou rovinu průmětny. Například projekce na rovinu os $x, z$ je reprezentována maticí $\operatorname{diag}(1, 0, 1)$. To odpovídá pozorovateli, který dívá ze směru osy $y$.

Výsledný obrázek pro vykreslení vzniknul pomocí několika lineárních transformací a celý postup jde tedy reprezentovat maticově součinem příslušných matic lineárních transformací.

</div>

### Shrnutí ke kapitole 6

Lineární zobrazení mezi vektorovými prostory zachovává strukturu lineárních kombinací: lineární kombinaci vektorů zobrazuje na tutéž lineární kombinaci jejich obrazů. K zadání lineárního zobrazení stačí uvést, kam se zobrazí vektory nějaké báze, a to plně určuje i obrazy ostatních vektorů i tím celého prostoru.

U aritmetických prostorů (typu $\mathbb{T}^n$) se lineární zobrazení dá vyjádřit maticově jako $x \mapsto Ax$. Každá matice tedy odpovídá nějakému lineárnímu zobrazení a naopak každé lineární zobrazení má maticové vyjádření. Tato dvojakost je zcela klíčová, protože mnoho problémů lze nahlížet algebraicky (operace s maticí $A$) nebo geometricky (pomocí lineárního zobrazení $x \mapsto Ax$). Řada vlastností lineárního zobrazení $x \mapsto Ax$ pak opět souvisí s vlastnostmi matice $A$:

- skládání zobrazení odpovídá maticovému součinu,
- zobrazení je prosté právě tehdy, když v jádru matice je pouze $o$,
- hodnost matice udává dimenzi obrazu,
- atp.

U obecných prostorů je situace trochu složitější, ale věci tam fungují podobně. Jenom se musí pracovat se souřadnicemi namísto vektorů samotných. Souřadnice jsou vektory z prostoru $\mathbb{T}^n$, tudíž výše zmíněné postřehy lze přizpůsobit na obecný případ. Při práci v souřadnicích se pak hojně využije matice přechodu, která převádí souřadnice v jedné bázi na souřadnice v jiné bázi; tedy změnu souřadného systému lze opět efektivně reprezentovat maticově.

Lineární zobrazení, které je bijekcí, se nazývá isomorfismus. Isomorfismy odpovídají regulárním maticím; proto také k nim (isomorfismům i maticím) existují inverze. Prostory, mezi kterými existuje isomorfismus, pak nazýváme isomorfní. Isomorfní prostory jsou z pohledu lineární algebry vlastně skoro stejné — mají jiné prvky, ale chovají se stejně. Mají také stejnou dimenzi, ale toto pozorování platí i naopak: Všechny $n$-dimenzionální prostory nad stejným tělesem jsou vzájemně isomorfní. To nám umožňuje nad každým prostorem snadno pracovat jako kdyby to byl prostor $\mathbb{T}^n$.

---

## Kapitola 7 — Afinní podprostory

Toto je letmý úvod do afinních podprostorů. Vektorové prostory a podprostory jsou omezeny tím, že musí obsahovat nulový vektor. Afinní podprostory zobecňují pojem podprostoru a vyhýbají se této restrikci. Afinním podprostorem v $\mathbb{R}^3$ tak může být jakákoli přímka či rovina, ne jenom ta procházející počátkem.

### 7.1 Základní pojmy

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Definice 7.1 — Afinní podprostor)</span></p>

Buď $V$ vektorový prostor nad $\mathbb{T}$. Pak *afinní podprostor* je jakákoli množina $M \subseteq V$ tvaru

$$M = U + a = \lbrace u + a;\ u \in U \rbrace,$$

kde $a \in V$ a $U$ je vektorový podprostor $V$.

</div>

Afinní podprostor (používá se i pojem afinní prostor či afinní množina se stejným významem) je tedy jakýkoli podprostor $U$ „posunutý" nějakým vektorem $a$.

Protože $o \in U$, je $a \in M$. Tento reprezentant $a$ není jednoznačný, můžeme zvolit libovolný vektor z $M$. Naopak, podprostor $U$ je u každého afinního podprostoru určený jednoznačně (problém 7.1).

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 7.2)</span></p>

Buď $V$ vektorový prostor. Každý jeho vektorový podprostor $U$ je zároveň jeho afinním podprostorem, neboť lze volit $a = o$, čímž $U = U + o$ je tvaru afinního podprostoru.

Dále, pro každý vektor $a \in V$ je množina $\lbrace a \rbrace$ jednoprvkový afinní podprostor ve $V$. Ten dostaneme volbou $U := \lbrace o \rbrace$, protože potom $U + a = \lbrace a \rbrace$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Poznámka 7.3 — Dláždění afinními podprostory)</span></p>

Buď $V$ vektorový prostor a $U$ jeho podprostor. Pak afinní podprostory tvaru $U + a$, $U + a'$ jsou buď shodné či disjunktní. Navíc každý vektor $v \in V$ leží v nějakém afinním podprostoru tohoto tvaru, například v afinním podprostoru $U + v$. Tudíž prostor $V$ lze rozložit na disjunktní sjednocení afinních podprostorů tvaru $U + a$ pro vhodné volby vektorů $a$.

</div>

#### Afinní kombinace

Zatímco vektorové podprostory jsou takové množiny vektorů, které jsou uzavřené na lineární kombinace, afinní podprostory jsou takové množiny vektorů, které jsou uzavřené na tzv. afinní kombinace.

*Afinní kombinace* dvou vektorů $x, y \in V$ (prostor nad tělesem $T$) je výraz (vektor) $\alpha x + (1 - \alpha)y$, kde $\alpha \in \mathbb{T}$. Afinní kombinaci lze přepsat do tvaru $\alpha x + (1 - \alpha)y = y + \alpha(x - y)$, což je parametrický popis přímky s bodem $y$ a směrnicí $x - y$. Jinými slovy, afinní podprostor s každými dvěma body musí obsahovat i přímku, která jimi prochází.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Věta 7.4 — Charakterizace afinního podprostoru)</span></p>

Buď $V$ vektorový prostor nad tělesem $\mathbb{T}$ charakteristiky různé od 2, a buď $\emptyset \neq M \subseteq V$. Pak $M$ je afinní podprostor právě tehdy, když pro každé $x, y \in M$ a $\alpha \in \mathbb{T}$ platí $\alpha x + (1 - \alpha)y \in M$.

</div>

*Důkaz.* Implikace „$\Rightarrow$": Nechť $M$ je tvaru $M = U + a$. Buď $x, y \in M$, tedy jsou tvaru $x = u + a$, $y = v + a$, kde $u, v \in U$. Potom $\alpha x + (1 - \alpha)y = \alpha(u + a) + (1 - \alpha)(v + a) = \alpha u + (1 - \alpha)v + a \in U + a = M$.

Implikace „$\Leftarrow$": Ukážeme, že stačí zvolit $a \in M$ libovolně pevně a $U := M - a = \lbrace x - a;\ x \in M \rbrace$. Musíme ověřit, že $M = U + a$ a že $U$ je vektorový podprostor. Rovnost $M = U + a$ je vidět z definice $U$, takže se zaměříme na druhou část a ukážeme $o \in U$ a uzavřenost $U$ na násobky a součty. Zřejmě $o \in U$.

Uzavřenost na násobky: Buď $\alpha \in \mathbb{T}$ a $u \in U$, tedy $u = x - a$ pro nějaké $x \in M$. Pak $\alpha u = \alpha(x - a) = (\alpha x + (1 - \alpha)a) - a \in M - a = U$, neboť $\alpha x + (1 - \alpha)a$ je afinní kombinace vektorů $x, a \in M$.

Uzavřenost na součty: Buďte $u, u' \in U$, tedy jsou tvaru $u = x - a$, $u' = x' - a$ pro nějaké $x, x' \in M$. Jejich součtem dostaneme $u + u' = (x - a) + (x' - a) = (x + x' - a) - a$. Stačí ukázat, že $x + x' - a \in M$. Protože $x, x' \in M$, také jejich afinní kombinace $\frac{1}{2}x + \frac{1}{2}x' \in M$. Protože $(\frac{1}{2}x + \frac{1}{2}x'), a \in M$, také jejich afinní kombinace $2(\frac{1}{2}x + \frac{1}{2}x') + (1 - 2)a = x + x' - a \in M$.

Implikace „$\Rightarrow$" platí vždy, ale obrácená implikace nemusí platit nad tělesem charakteristiky 2. Stačí vzít za příklad prostor $\mathbb{Z}_2^n$ nad $\mathbb{Z}_2$, v němž je každá množina vektorů uzavřená na afinní kombinace dvou vektorů.

Větu můžeme zobecnit i na tělesa charakteristiky 2, musíme ale rozšířit pojem afinní kombinace na větší počet vektorů. *Afinní kombinace* vektorů $x_1, \ldots, x_n \in V$ je výraz (vektor)

$$\sum_{i=1}^{n} \alpha_i x_i, \quad \text{kde } \alpha_i \in \mathbb{T}, \quad \sum_{i=1}^{n} \alpha_i = 1.$$

Jedná se o takovou lineární kombinaci, u které je součet koeficientů roven 1. Pro dva vektory dostáváme původní definici. Geometrická interpretace pro tři vektory (body) říká, že jejich afinní kombinace popisují rovinu, která je těmito body určena.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Tvrzení 7.5)</span></p>

Buď $V$ vektorový prostor nad $\mathbb{T}$ a buď $\emptyset \neq M \subseteq V$. Pak $M$ je afinní podprostor právě tehdy, když $M$ je uzavřená na afinní kombinace.

</div>

*Důkaz.* Analogický důkazu věty 7.4. Důkaz uzavřenosti množiny $U$ na součty vyplývá přímo z toho, že $x + x' - a \in M$, protože se jedná o afinní kombinaci tří vektorů $x, x', a$ (jejich koeficienty se sečtou na $1 + 1 - 1 = 1$). Proto není potřeba nikde dělit dvěma a lze uvažovat libovolné těleso.

Z důkazu vidíme, že stačí, aby množina $M$ byla uzavřená na afinní kombinace tří vektorů. Pak už je uzavřená na afinní kombinace libovolného konečného počtu vektorů.

#### Afinní podprostory a soustavy lineárních rovnic

Je velmi těsný vztah mezi afinními podprostory a množinou řešení soustav lineárních rovnic.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Věta 7.6 — Soustavy lineárních rovnic a afinní podprostory)</span></p>

Množina řešení soustavy rovnic $Ax = b$ je prázdná nebo afinní. Je-li neprázdná, můžeme tuto množinu řešení vyjádřit ve tvaru $\operatorname{Ker}(A) + x_0$, kde $x_0$ je jedno libovolné řešení soustavy.

</div>

*Důkaz.* Pokud $x_1$ je řešením, pak lze psát $x_1 = x_1 - x_0 + x_0$. Stačí ukázat, že $x_1 - x_0 \in \operatorname{Ker}(A)$. Dosazením $A(x_1 - x_0) = Ax_1 - Ax_0 = b - b = o$. Tedy $x_1 \in \operatorname{Ker}(A) + x_0$. Naopak, je-li $x_2 \in \operatorname{Ker}(A)$, pak $x_2 + x_0$ je řešením soustavy, neboť $A(x_2 + x_0) = Ax_2 + Ax_0 = o + b = b$.

Ukážeme že platí i obrácená implikace, tedy každý afinní podprostor prostoru $\mathbb{T}^n$ nad $\mathbb{T}$ lze popsat pomocí soustavy rovnic.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Tvrzení 7.7 — Afinní podprostory a soustavy lineárních rovnic)</span></p>

Buď $U + a$ afinní podprostor prostoru $\mathbb{T}^n$ nad $\mathbb{T}$. Pak existuje matice $A \in \mathbb{T}^{m \times n}$ a vektor $b \in \mathbb{T}^m$ takové, že množina řešení soustavy lineárních rovnic $Ax = b$ je rovna $U + a$.

</div>

*Důkaz.* Buď $v_1, \ldots, v_k \in \mathbb{T}^n$ báze podprostoru $U$. Sestavíme matici $C \in \mathbb{T}^{k \times n}$, jejíž řádky jsou vektory $v_1, \ldots, v_k$. Dimenze jejího jádra je $\dim \operatorname{Ker}(C) = n - \operatorname{rank}(C) = n - k$. Buď $w_1, \ldots, w_{n-k}$ báze $\operatorname{Ker}(C)$. Platí tedy $Cw_j = o$, čili speciálně pro řádky matice $C$ dostaneme $v_i^T w_j = 0$ pro $i = 1, \ldots, k$, $j = 1, \ldots, n - k$. Nyní sestavíme matici $A \in \mathbb{T}^{(n-k) \times n}$ tak, že její řádky jsou tvořeny vektory $w_1, \ldots, w_{n-k}$. Dimenze jejího jádra je $\dim \operatorname{Ker}(A) = n - \operatorname{rank}(A) = n - (n - k) = k$. Protože jsou vektory $v_1, \ldots, v_k$ lineárně nezávislé a je jich správný počet, tvoří bázi $\operatorname{Ker}(A)$. Tudíž $\operatorname{Ker}(A) = U$. Zbývá určit vektor $b$, aby vektor $a$ byl řešením soustavy $Ax = b$. Stačí tedy zvolit $b := Aa$. Podle věty 7.6 je množina řešení soustavy $Ax = b$ rovna $\operatorname{Ker}(A) + a = U + a$.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 7.8 — Soustava lineárních rovnic při změně pravé strany)</span></p>

Věta 7.6 dává následující geometrický pohled na soustavy rovnic při perturbaci pravé strany. Nechť $Ax = b$ je řešitelná, tedy popisuje afinní podprostor $\operatorname{Ker}(A) + x_0$, kde $x_0'$ je jedno vybrané řešení. Změníme-li pravou stranu soustavy $b$ na $b'$, pak buďto soustava přestane mít řešení, nebo se afinní podprostor posune na $\operatorname{Ker}(A) + x_0'$, kde $x_0'$ je jedno vybrané řešení. Jsou-li řádky matice $A$ lineárně nezávislé, pak soustava je řešitelná pro jakoukoli pravou stranu, a tudíž se množina řešení při změně pravé strany pouze posouvá nějakým směrem. Jsou-li řádky matice $A$ lineárně závislé, pak pro některá $b$ je soustava řešitelná a pro některá není. Pro ty hodnoty $b$, pro něž je soustava řešitelná, je opět množina řešení stejná až na posunutí.

Pro konkrétnost uvažujme soustavu lineárních rovnic s obecnou pravou stranou

$$(A \mid b) = \begin{pmatrix} 1 & 1 & 3 & b_1 \\ 2 & 1 & 1 & b_2 \end{pmatrix}.$$

Řádky matice $A$ jsou lineárně nezávislé a pro každé $b = (b_1, b_2)^T$ má množina řešení tvar $\operatorname{span}\lbrace (2, -5, 1)^T \rbrace + (-b_1 + b_2, 2b_1 - b_2, 0)^T$. Je to tedy přímka vždy se stejnou směrnicí.

Nyní uvažujme soustavu lineárních rovnic s lineárně závislými řádky matice $A$

$$(A \mid b) = \begin{pmatrix} 1 & 2 & 3 & b_1 \\ 2 & 4 & 6 & b_2 \end{pmatrix}.$$

Pokud $b_2 \neq 2b_1$, řešení neexistuje. Pokud $b_2 = 2b_1$, množina řešení je rovina popsaná rovnicí $x_1 + 2x_2 + 3x_3 = b_1$ a její normála nezávisí na pravé straně.

</div>

#### Dimenze afinního podprostoru

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Definice 7.9 — Dimenze afinního podprostoru)</span></p>

*Dimenze* afinního podprostoru $M = U + a$ je definována jako $\dim(M) := \dim(U)$.

</div>

Protože každý vektorový podprostor prostoru $V$ je zároveň jeho afinním podprostorem, definice tedy zobecňuje pojem dimenze, zavedený pro vektorové prostory. Definice přirozeně zavádí dimenzi bodu jako nula, dimenzi přímky v $\mathbb{R}^n$ jako jedna a dimenzi roviny jako dva.

Také umožňuje definovat *přímku* $p$ v libovolném vektorovém prostoru $V$ nad $\mathbb{T}$ jakožto afinní podprostor dimenze jedna. Jinými slovy, $p = \operatorname{span}\lbrace v \rbrace + a$, kde $a, v \in V$ a $v \neq o$. Odsud dostáváme i známý parametrický popis přímky $p = \lbrace \alpha v + a;\ \alpha \in \mathbb{T} \rbrace$.

*Nadrovinou* v prostoru dimenze $n$ rozumíme pak libovolný afinní podprostor dimenze $n - 1$. Tedy například v $\mathbb{R}^2$ jsou to přímky, v $\mathbb{R}^3$ roviny, atd.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 7.10)</span></p>

Množina $\lbrace e^x + \alpha \sin x;\ \alpha \in \mathbb{R} \rbrace$ je přímka v prostoru funkcí $\mathcal{F}$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 7.11)</span></p>

Pro jakékoli $a \in \mathbb{R}^n \setminus \lbrace o \rbrace$ a $b \in \mathbb{R}$ je množina popsaná rovnicí $a^T x = b$ nadrovinou v $\mathbb{R}^n$. A naopak, každá nadrovina v $\mathbb{R}^n$ se dá popsat rovnicí $a^T x = b$ pro určité $a \in \mathbb{R}^n \setminus \lbrace o \rbrace$ a $b \in \mathbb{R}$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Tvrzení 7.12)</span></p>

Buď $A \in \mathbb{T}^{m \times n}$, $b \in \mathbb{T}^m$. Je-li množina řešení soustavy rovnic $Ax = b$ neprázdná, pak ji tvoří afinní podprostor dimenze $n - \operatorname{rank}(A)$.

</div>

*Důkaz.* Podle tvrzení 7.6 jde množina řešení vyjádřit ve tvaru $\operatorname{Ker}(A) + x_0$, kde $x_0$ je jedno libovolné řešení soustavy. Její dimenze je tedy rovna dimenzi jádra, což je podle věty 5.72 rovno $\dim \operatorname{Ker}(A) = n - \operatorname{rank}(A)$.

#### Afinní nezávislost

Lineární nezávislost vektorů $x_1, \ldots, x_n$ znamenala, že podprostor, který tyto vektory generují, se nedá vygenerovat nějakou vlastní podmnožinou těchto vektorů. Žádný z nich není v jistém smyslu zbytečný. Chtěli bychom analogickou vlastnost mít i pro afinní podprostory, tedy umět charakterizovat nejmenší množinu vektorů, jenž jednoznačně určují daný afinní podprostor. Toto vede na pojem afinní nezávislost.

Afinní podprostor jsme definovali jako $U + a$, tedy vektorový podprostor $U$ posunutý o vektor $a$. Nabízí se tedy definovat afinní nezávislost vektorů tak, že je posuneme zpět a po výsledných vektorech budeme požadovat lineární nezávislost.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Definice 7.13 — Afinní nezávislost)</span></p>

Vektory $x_0, x_1, \ldots, x_n$ vektorového prostoru jsou *afinně nezávislé*, pokud $x_1 - x_0, \ldots, x_n - x_0$ jsou lineárně nezávislé. V opačném případě vektory nazýváme *afinně závislé*.

</div>

Vektory $x_0, x_1, \ldots, x_n \in V$ jednoznačně určují nejmenší (ve smyslu inkluze) afinní podprostor, který je obsahuje. Označme ho $M = U + x_0$. Vektory $x_1 - x_0, \ldots, x_n - x_0$ jsou generátory podprostoru $U$. Tyto generátory tvoří bázi $U$ právě tehdy, když vektory $x_0, x_1, \ldots, x_n$ jsou afinně nezávislé. Pokud bychom jakýkoli z vektorů $x_0, x_1, \ldots, x_n$ odstranili, tak nevygenerujeme celý afinní podprostor $M$, ale jenom jeho část.

Není těžké nahlédnout, že afinní nezávislost nezávisí na pořadí vektorů, a tedy ani na volbě $x_0$ (dokažte!).

Afinní nezávislost navíc umožňuje jednoduše formalizovat to, co známe pod pojmem „Mějme body v obecné poloze": Množina bodů $x_1, \ldots, x_m \in \mathbb{R}^n$ je v obecné poloze, když každá její podmnožina velikosti nanejvýš $n + 1$ je afinně nezávislá. Tedy například v rovině $\mathbb{R}^2$ jsou dané body v obecné poloze pokud žádné tři neleží na společné přímce.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 7.14)</span></p>

Vektory $(1, 1)^T, (2, 2)^T, (1, 2)^T \in \mathbb{R}^2$ jsou sice lineárně závislé, ale afinně nezávislé.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 7.15)</span></p>

Dva různé body v $\mathbb{R}^n$ jsou afinně nezávislé a afinní podprostor, který generují, je přímka. Nicméně, tři body na přímce už jsou afinně závislé, protože přímka je jednoznačně určena jen dvěma body.

</div>

#### Souřadnice v afinním podprostoru

Buď $M = U + a$ afinní podprostor a $B = \lbrace v_1, \ldots, v_n \rbrace$ báze $U$. Pak každé $x \in M$ se dá jednoznačně zapsat ve tvaru $x = a + \sum_{i=1}^{n} \alpha_i v_i$. Tedy systém vektorů $S = \lbrace a, v_1, \ldots, v_n \rbrace$ lze považovat za *souřadný systém* a vektor $[x]_S := [x - a]_B = (\alpha_1, \ldots, \alpha_n)^T$ za příslušné *souřadnice*.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 7.16)</span></p>

Buď $v = (2, 1)^T$, $a = (1, 2)^T$ a uvažujme přímku $\operatorname{span}\lbrace v \rbrace + a$. Uvažujme dále souřadný systém $S = \lbrace a, v \rbrace$. Potom bod $x = (5, 4)^T$ lze vyjádřit jako $x = a + 2v$, a proto jeho souřadnice jsou $[x]_S = (2)$.

Uvažujme nyní jiný vektor $a' = (-1, 1)^T$. Potom se vyjádření vektoru $x$ změní na $x = a' + 3v$, a proto jeho souřadnice v systému $S' = \lbrace a', v \rbrace$ budou $[x]_{S'} = (3)$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Tvrzení 7.17)</span></p>

Buď $M = U + a$ afinní podprostor a $B = \lbrace v_1, \ldots, v_n \rbrace$, $B' = \lbrace v_1', \ldots, v_n' \rbrace$ dvě báze $U$.

1. Pro dva dané souřadné systémy $S = \lbrace a, v_1, \ldots, v_n \rbrace$ a $S' = \lbrace a, v_1', \ldots, v_n' \rbrace$ máme

$$[x]_{S'} = {}_{B'}[id]_B \cdot [x]_S, \quad \forall x \in U + a.$$

2. Pro dva dané souřadné systémy $S = \lbrace a, v_1, \ldots, v_n \rbrace$ a $S' = \lbrace a', v_1', \ldots, v_n' \rbrace$ máme

$$[x]_{S'} = [a - a']_{B'} + {}_{B'}[id]_B \cdot [x]_S, \quad \forall x \in U + a.$$

</div>

K přechodu mezi souřadnými systémy pak můžeme použít naši známou matici přechodu přesně tak, jak jsme byli zvyklí. V případě, že změníme i vektor $a$, objeví se navíc konstantní aditivní člen.

#### Vztah afinních podprostorů

Afinní podprostory $U + a$, $W + b$ jsou *rovnoběžné*, pokud $U \subseteq W$ nebo $W \subseteq U$; *různoběžné*, pokud nejsou rovnoběžné a mají neprázdný průnik; a *mimoběžné*, pokud nejsou rovnoběžné a mají prázdný průnik.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 7.18)</span></p>

Buď $v$ libovolný vektor vektorového prostoru $V$. Pak afinní podprostor $\lbrace v \rbrace$ je rovnoběžný s každým afinním podprostorem $V$. Stejnou vlastnost má i celý prostor $V$.

</div>

#### Afinní zobrazení

Buď $g \colon U \to V$ lineární zobrazení a mějme pevný vektor $b \in V$. Potom *afinní zobrazení* má tvar $f(u) = g(u) + b$. Jednoduchým příkladem afinního zobrazení je posunutí, tedy zobrazení $g \colon V \to V$ s popisem $f(x) = x + b$, kde $b \in V$ je pevné.

Afinní zobrazení nemusí zobrazovat nulový vektor v $U$ na nulový vektor ve $V$, protože jsou obrazy posunuté o aditivní člen $b$.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 7.19)</span></p>

Buď $U + a$ afinní podprostor dimenze $k$ v prostoru $V$, a buď $S$ souřadný systém v $U + a$. Pak zobrazení $f(v) = [v]_S$ je afinní zobrazení zobrazující isomorfně $U + a$ na $\mathbb{T}^k$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Tvrzení 7.20 — Vlastnosti afinního zobrazení)</span></p>

1. Obraz afinního podprostoru při afinním zobrazení je afinní podprostor.
2. Složením dvou afinních zobrazení dostaneme opět afinní zobrazení.
3. Buď $f \colon U \to V$ lineární zobrazení a vektor $v \in V$. Pak úplný vzor vektoru $v$

$$f^{-1}(v) := \lbrace u \in U;\ f(u) = v \rbrace$$

je buďto prázdná množina, nebo afinní podprostor v $U$.

</div>

*Důkaz.*

1. Buď $f \colon U \to V$ afinní zobrazení ve tvaru $f(u) = g(u) + b$, kde $g \colon U \to V$ je lineární a $b \in V$. Pak obraz afinního podprostoru $W + a \subseteq U$ je $f(W + a) = g(W + a) + b = g(W) + g(a) + b$. Jedná se tedy o afinní podprostor ve $V$, vzniklý posunem podprostoru $g(W)$ ve směru vektoru $g(a) + b$.
2. Mějme $f_1 \colon U \to V$, $f_2 \colon V \to W$ afinní zobrazení ve tvaru $f_1(u) = g_1(u) + b_1$, $f_2(v) = g_2(v) + b_2$, kde $g_1 \colon U \to V$, $g_2 \colon V \to W$ jsou lineární a $b_1 \in V$, $b_2 \in W$. Pak složené zobrazení má tvar $(f_2 \circ f_1)(u) = f_2(f_1(u)) = g_2(g_1(u) + b_1) + b_2 = g_2(g_1(u)) + g_2(b_1) + b_2 = (g_2 \circ g_1)(u) + g_2(b_1) + b_2$. Jedná se tedy opět o afinní zobrazení, vzniklé z lineárního zobrazení $g_2 \circ g_1$ posunem o aditivní člen $g_2(b_1) + b_2$.
3. Buďte $U, V$ prostory nad tělesem $\mathbb{T}$ a buďte $u_1, \ldots, u_n \in f^{-1}(v)$. Uvažujme jejich afinní kombinaci $\sum_{i=1}^{n} \alpha_i u_i$, kde $\alpha_1, \ldots, \alpha_n \in \mathbb{T}$ a $\sum_{i=1}^{n} \alpha_i = 1$. Pak $f(\sum_{i=1}^{n} \alpha_i u_i) = \sum_{i=1}^{n} \alpha_i f(u_i) = \sum_{i=1}^{n} \alpha_i v = v$. Tudíž $\sum_{i=1}^{n} \alpha_i u_i \in f^{-1}(v)$, což ukazuje, že množina $f^{-1}(v)$ je uzavřená na afinní kombinace.

Bod (3) tvrzení 7.20 má analogii s řešením soustav lineárních rovnic. Uvažujme lineární zobrazení $f \colon \mathbb{R}^n \to \mathbb{R}^m$ vyjádřené maticově $f(x) = Ax$ a mějme dané $b \in \mathbb{R}^m$. Potom hledat všechna řešení soustavy $Ax = b$ vlastně znamená najít úplný vzor vektoru $b$, tedy množinu

$$f^{-1}(b) = \lbrace x \in \mathbb{R}^n;\ f(x) = b \rbrace = \lbrace x \in \mathbb{R}^n;\ Ax = b \rbrace.$$

Z věty 7.6 pak víme, že množina řešení soustavy $Ax = b$ je prázdná nebo afinní podprostor. Ještě jiný pohled na bod (3) tvrzení 7.20 je pomocí jádra lineárního zobrazení. Podobně jako ve větě 7.6 můžeme úplný vzor vektoru $v$ vyjádřit jako afinní podprostor $\operatorname{Ker}(f) + u$, kde $u \in U$ je jeden pevný vzor vektoru $v$.

### 7.2 Aplikace

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 7.21 — Rovnice ano, ale nerovnice?)</span></p>

V minulých kapitolách jsme studovali soustavy lineárních rovnic $Ax = b$, tudíž je přirozená otázka zabývat se i soustavou lineárních nerovnic $Ax \le b$. Nerovnost mezi vektory znamená nerovnost v každé složce, tedy $(Ax)_i \le b_i$ pro všechna $i$. Dále, omezíme se jen na těleso $\mathbb{R}$, kde máme definováno uspořádání.

Zatímco jedna rovnice vytyčuje v prostoru nadrovinu a soustava rovnic pak nějaký afinní podprostor, tak jedna nerovnice vymezuje v prostoru poloprostor a soustava nerovnic pak průnik poloprostorů, což je *konvexní polyedr (mnohostěn)*.

Čtyřúhelník se skládá ze čtyř vrcholů, čtyř hran a vnitřku. Vrcholy leží v průniku nadrovin odpovídajících soustavě rovnic a podle věty 7.6 je to afinní podprostor dimenze nula. Hrana spojující dva sousední vrcholy leží na jednorozměrném afinním podprostoru. Analogicky charakterizujeme další vrcholy a hrany.

Podobným způsobem pokračujeme ve vyšších dimenzích. Například trojdimenzionální polyedry, jako je krychle, osmistěn atp. mají následující strukturu. Vrcholy jsou nula-dimenzionální afinní podprostory určené průnikem tří nadrovin, tj. popsané třemi rovnicemi. Hrany leží v jednodimenzionálním afinním podprostoru popsaném soustavou dvou rovnic. A konečně stěny leží v nadrovině, tedy ve dvoudimenzionálním afinním podprostoru, který je určen jednou rovnicí.

Konvexními polyedry se více zabývá například obor *lineární programování*. Ten zkoumá nejen konvexní polyedry, ale také nad nimi řeší optimalizační úlohy typu

$$\min\ c^T x \quad \text{za podmínek } Ax \le b,$$

kde $c \in \mathbb{R}^n$, $A \in \mathbb{R}^{m \times n}$ a $b \in \mathbb{R}^m$ jsou dané a $x \in \mathbb{R}^n$ je vektor proměnných. Lineární programování tedy hledá minimum lineární funkce na konvexním polyedru. Tento problém je základní úlohou optimalizace a objevuje se téměř ve všech úlohách, které s optimalizací souvisí: například v rozvrhování a plánování, dopravních úlohách (nalezení nejkratší cesty) nebo při hledání optimální strategie v teorii her.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 7.22 — Afinní zobrazení a fraktály)</span></p>

Fraktál je soběpodobný geometrický útvar na první pohled složitého tvaru. Ukážeme na příkladu, že i poměrně složitý fraktál může mít jednoduchý popis pomocí afinních zobrazení.

Pomocí čtyř afinních zobrazení dokážeme v rovině vykreslit složitý fraktál. Začneme v počátku a s danými pravděpodobnostmi uvažujeme přechod podle příslušného afinního zobrazení.

$$T_1(x, y) = \begin{pmatrix} 0.86 & 0.03 \\ -0.03 & 0.86 \end{pmatrix} \begin{pmatrix} x \\ y \end{pmatrix} + \begin{pmatrix} 0 \\ 1.5 \end{pmatrix} \quad \text{s pravděpodobností 0.83}$$

$$T_2(x, y) = \begin{pmatrix} 0.2 & -0.25 \\ 0.21 & 0.23 \end{pmatrix} \begin{pmatrix} x \\ y \end{pmatrix} + \begin{pmatrix} 0 \\ 1.5 \end{pmatrix} \quad \text{s pravděpodobností 0.08}$$

$$T_3(x, y) = \begin{pmatrix} -0.15 & 0.27 \\ 0.25 & 0.26 \end{pmatrix} \begin{pmatrix} x \\ y \end{pmatrix} + \begin{pmatrix} 0 \\ 0.45 \end{pmatrix} \quad \text{s pravděpodobností 0.08}$$

$$T_4(x, y) = \begin{pmatrix} 0 & 0 \\ 0 & 0.17 \end{pmatrix} \begin{pmatrix} x \\ y \end{pmatrix} + \begin{pmatrix} 0 \\ 0 \end{pmatrix} \quad \text{s pravděpodobností 0.01}$$

Navštívené body postupně vykreslí tzv. Barnsleyho fraktál ve tvaru listu kapradiny.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 7.23 — Stewartova–Goughova platforma v robotice)</span></p>

Stewartova–Goughova platforma je tzv. paralelní manipulátor v oboru kinematické robotiky. Pevná základna je připevněna několika (většinou šesti) pohyblivými rameny k mobilní plošině. Tyto platformy se využívají jako manipulátory, v simulacích (např. letů), nebo třeba v biomechanice kloubů k ověřování implantátů mimo lidské tělo.

Základna i mobilní plošina mají své vlastní souřadné systémy, mezi kterými můžeme přecházet pomocí afinního zobrazení. Například jsou-li $x = (x_1, x_2, x_3)^T$ souřadnice bodu v systému plošiny, pak souřadnice vůči základně získáme jako $x' = Px + c$, kde $P$ matice reprezentující naklonění a $c$ je nějaký pevný vektor reprezentující posun. Pochopitelně, $P$ a $c$ nejsou pevné, ale závisí na míře natažení pohyblivých ramen. Navíc se dá ukázat, že matice $P$ závisí pouze na třech parametrech, protože systém plošiny vzhledem k základně je pouze natočený a není nijak deformovaný (natáhnutý, zkosený atp.).

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 7.24 — Lineární klasifikátor a neuronové sítě)</span></p>

Mějme data reprezentovaná vektory $v_1, \ldots, v_m \in \mathbb{R}^n$ a pro každou hodnotu víme, zda patří do skupiny A či B. Nechť $v_i$ patří do skupiny A pro $i \in \mathcal{A}$ a do skupiny B pro $i \in \mathcal{B}$. Chceme sestrojit klasifikátor, který bude umět pro novou hodnotu $v \in \mathbb{R}^n$ automaticky rozhodnout do které skupiny patří. Jednoduchý klasifikátor můžeme sestrojit na základě lineárního separátoru. Jeho podstata spočívá v tom sestrojit nadrovinu $a^T x = b$ takovou, aby vektory skupiny A byly v jedné polorovině a vektory skupiny B ve druhé.

Matematicky popsáno, hledáme $a \in \mathbb{R}^n$, $b \in \mathbb{R}$ tak, aby

$$a^T v_i < b \quad \forall i \in \mathcal{A}, \qquad a^T v_i > b \quad \forall i \in \mathcal{B}.$$

Jestliže takovou nadrovinu najdeme, klasifikace nového údaje $v \in \mathbb{R}^n$ funguje jednoduše. Pokud $a^T v < b$, pak hodnotu $v$ považujeme za člena skupiny A a v opačném případě za člena skupiny B. Pokud lineární separátor nenajdeme, je situace trochu složitější. Jedna z možností, jak se s tím vypořádat, je vnořit data do prostoru vyšší dimenze, kde je větší šance na úspěch.

Lineární klasifikátory se využívají mj. v neuronových sítích. Je to jeden ze způsobů, jaký perceptronové algoritmy učení využívají k hledání váhových koeficientů vazeb neuronů.

</div>

### Shrnutí ke kapitole 7

Podprostor vektorového prostoru musí procházet počátkem. Pokud však podprostor posuneme ve směru nějakého vektoru, dostáváme nový objekt — afinní podprostor. Zatímco vektorové podprostory jsou uzavřené na lineární kombinace, afinní podprostory jsou (za obecných předpokladů) uzavřené na afinní kombinace. Důležitý je vztah se soustavami lineárních rovnic — množina řešení soustavy $Ax = b$ je afinním podprostorem, a naopak každý afinní podprostor se dá vyjádřit jako řešení určité soustavy.

Řada pojmů a vlastností z vektorových prostorů se přirozeně přenese na afinní podprostory. Tudíž snadno zavedeme pojmy jako (afinní) nezávislost, báze, souřadnice či dimenze. Afinní zobrazení pak je zobrazení, které má tvar lineárního zobrazení s konstantním aditivním členem; u prostorů typu $\mathbb{T}^n$ má tvar $x \mapsto Ax + b$. Toto zobrazení má opět podobné vlastnosti jako lineární zobrazení, jenom přeložené do světa afinních podprostorů.

---

## Kapitola 8 — Skalární součin

Podle definice 5.2 musíme umět vektory sčítat a násobit skalárem, ale násobení vektorů mezi sebou nebylo zatím požadováno. Nicméně zavedení skalárního součinu vektorů umožní také přirozeně zavést pojem kolmosti, velikosti a vzdálenosti vektorů (a tím i limity) atd.

### 8.1 Skalární součin a norma

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 8.1 — Motivační)</span></p>

Standardní skalární součin (str. 41) vektorů $x, y \in \mathbb{R}^n$ je definován jako

$$x^T y = \sum_{i=1}^{n} x_i y_i$$

a pomocí něj snadno vyjádříme velikost vektoru nebo úhel mezi dvěma vektory. Eukleidovská norma (velikost) vektoru $x \in \mathbb{R}^n$ je definována jako $\|x\| = \sqrt{x^T x} = \sqrt{\sum_{i=1}^{n} x_i^2}$. Pochopitelně zde platí $x^T x \ge 0$. Jediný vektor, který má nulovou normu, je nulový vektor.

Geometricky vyjadřuje skalární součin vztah

$$x^T y = \|x\| \cdot \|y\| \cdot \cos(\varphi),$$

kde $\varphi$ je úhel mezi vektory $x, y$. Tedy ze znalosti vektorů $x, y$ snadno vypočítáme úhel mezi nimi. Speciálně, $x, y$ jsou kolmé právě tehdy, když $x^T y = 0$.

Z definice je ihned vidět, že skalární součin je symetrický, tedy $x^T y = y^T x$. Skalární součin je také lineární funkcí v první i druhé složce, ale ne v obou zároveň. To znamená, že pro každé $x, x', y \in \mathbb{R}^n$ a $\alpha \in \mathbb{R}$ platí

$$(x + x')^T y = x^T y + x'^T y, \qquad (\alpha y)^T y = \alpha (x^T y)$$

(a symetricky pro druhou složku), ale obecně neplatí například $(x + x')^T(y + y') = x^T y + x'^T y'$.

</div>

Skalární součin (stejně jako grupu, vektorové prostory aj.) zavádíme axiomaticky, tedy výčtem vlastností, které má splňovat.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Definice 8.2 — Skalární součin nad $\mathbb{R}$)</span></p>

Buď $V$ vektorový prostor nad $\mathbb{R}$. Pak *skalární součin* je zobrazení $\langle \cdot, \cdot \rangle \colon V^2 \to \mathbb{R}$, splňující pro všechna $x, y, z \in V$ a $\alpha \in \mathbb{R}$:

1. $\langle x, x \rangle \ge 0$ a rovnost nastane pouze pro $x = 0$,
2. $\langle x + y, z \rangle = \langle x, z \rangle + \langle y, z \rangle$,
3. $\langle \alpha x, y \rangle = \alpha \langle x, y \rangle$,
4. $\langle x, y \rangle = \langle y, x \rangle$.

</div>

Vlastnost (1) vyžaduje uspořádání, proto jsme zavedli skalární součin nad tělesem reálných čísel. Vlastnosti (2) a (3) říkají, že skalární součin je lineární funkcí v první složce. Díky vlastnosti (4) je skalární součin symetrický, a proto je lineární funkcí i ve druhé složce. To znamená, že pro každé $x, y, z \in V$ a $\alpha, \beta \in \mathbb{R}$ platí

$$\langle x, \alpha y + \beta z \rangle = \alpha \langle x, y \rangle + \beta \langle x, z \rangle.$$

Pokud použijeme vlastnost (3) s hodnotou $\alpha = 0$, dostáváme $\langle o, x \rangle = \langle x, o \rangle = 0$, tedy násobení jakéhokoli vektoru s nulovým vektorem dá nulu.

Nyní rozšíříme definici i na těleso komplexních čísel. Připomeňme, že komplexně sdružené číslo k číslu $a + bi \in \mathbb{C}$ je definované jako $\overline{a + bi} = a - bi$. Vzhledem ke čtvrté vlastnosti dole je nutně $\langle x, x \rangle = \overline{\langle x, x \rangle}$, tudíž $\langle x, x \rangle$ je vždy reálné číslo a lze jej porovnávat s nulou.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Definice 8.3 — Skalární součin nad $\mathbb{C}$)</span></p>

Buď $V$ vektorový prostor nad $\mathbb{C}$. Pak *skalární součin* je zobrazení $\langle \cdot, \cdot \rangle \colon V^2 \to \mathbb{C}$, splňující pro všechna $x, y, z \in V$ a $\alpha \in \mathbb{C}$:

1. $\langle x, x \rangle \ge 0$ a rovnost nastane pouze pro $x = 0$,
2. $\langle x + y, z \rangle = \langle x, z \rangle + \langle y, z \rangle$,
3. $\langle \alpha x, y \rangle = \alpha \langle x, y \rangle$,
4. $\langle x, y \rangle = \overline{\langle y, x \rangle}$.

</div>

Druhá a třetí vlastnost opět říkají, že skalární součin je lineární funkcí v první složce. Jak je to s druhou?

$$\langle x, y + z \rangle = \overline{\langle y + z, x \rangle} = \overline{\langle y, x \rangle} + \overline{\langle z, x \rangle} = \langle x, y \rangle + \langle x, z \rangle,$$

$$\langle x, \alpha y \rangle = \overline{\langle \alpha y, x \rangle} = \overline{\alpha} \overline{\langle y, x \rangle} = \overline{\alpha} \langle x, y \rangle.$$

Ve druhé složce tedy komplexní skalární součin již není lineární.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 8.4 — Příklady standardních skalárních součinů)</span></p>

- V prostoru $\mathbb{R}^n$: standardní skalární součin $\langle x, y \rangle = x^T y = \sum_{i=1}^{n} x_i y_i$.
- V prostoru $\mathbb{C}^n$: standardní skalární součin $\langle x, y \rangle = x^T \overline{y} = \sum_{i=1}^{n} x_i \overline{y_i}$.
- V prostoru $\mathbb{R}^{m \times n}$: standardní skalární součin $\langle A, B \rangle = \sum_{i=1}^{m} \sum_{j=1}^{n} a_{ij} b_{ij}$.
- V $\mathcal{C}_{[a,b]}$, prostoru spojitých funkcí na intervalu $[a, b]$: standardní skalární součin $\langle f, g \rangle = \int_a^b f(x) g(x) \, dx$.

</div>

Výše zmíněné skalární součiny jsou pouze příklady možných zavedení součinů na daných prostorech; jako skalární součin mohou fungovat i jiné operace. Později, ve větě 11.18, popíšeme všechny skalární součiny v prostoru $\mathbb{R}^n$.

Je dobré si uvědomit, že zobrazení $\langle x, y \rangle = x^T y$ v prostoru $\mathbb{C}^n$ skalární součin netvoří, protože například pro vektor $x = (i, i)^T$ by bylo $\langle x, x \rangle = x^T x = -2$.

Nadále uvažujme vektorový prostor $V$ nad $\mathbb{R}$ či $\mathbb{C}$ se skalárním součinem. Nejprve ukážeme, že skalární součin umožňuje zavést normu, neboli velikost vektoru.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Definice 8.5 — Norma indukovaná skalárním součinem)</span></p>

*Norma indukovaná skalárním součinem* je definovaná $\|x\| := \sqrt{\langle x, x \rangle}$, kde $x \in V$.

</div>

Norma je dobře definovaná díky první vlastnosti z definice skalárního součinu, a je to vždy nezáporná hodnota. Pro standardní skalární součin v $\mathbb{R}^n$ dostáváme eukleidovskou normu $\|x\| = \sqrt{x^T x} = \sqrt{\sum_{i=1}^{n} x_i^2}$.

Jak jsme připomněli v příkladu 8.1, pro standardní skalární součin v $\mathbb{R}^n$ platí, že $x, y$ jsou kolmé právě tehdy, když $\langle x, y \rangle = 0$. V jiných vektorových prostorech takovýto geometrický náhled chybí, proto kolmost zavedeme právě pomocí vztahu $\langle x, y \rangle = 0$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Definice 8.6 — Kolmost)</span></p>

Vektory $x, y \in V$ jsou *kolmé*, pokud $\langle x, y \rangle = 0$. Značení: $x \perp y$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 8.7 — Příklady kolmých vektorů pro standardní skalární součiny)</span></p>

- V prostoru $\mathbb{R}^3$: $(1, 2, 3) \perp (1, 1, -1)$.
- V prostoru $\mathbb{R}^n$: $i$-tý řádek regulární matice $A \in \mathbb{R}^{n \times n}$ a $j$-tý sloupec matice $A^{-1}$ pro libovolné $i \neq j$.
- V prostoru $\mathcal{C}_{[-\pi, \pi]}$: $\sin x \perp \cos x \perp 1$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Věta 8.8 — Pythagorova)</span></p>

*Pokud $x, y \in V$ jsou kolmé, tak $\|x + y\|^2 = \|x\|^2 + \|y\|^2$.*

*Důkaz.* $\|x + y\|^2 = \langle x + y, x + y \rangle = \langle x, x \rangle + \underbrace{\langle x, y \rangle}\_{=0} + \underbrace{\langle y, x \rangle}\_{=0} + \langle y, y \rangle = \|x\|^2 + \|y\|^2$.

</div>

Poznamenejme, že nad $\mathbb{R}$ platí i opačná implikace, ale nad $\mathbb{C}$ obecně nikoli (viz problém 8.2).

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Věta 8.9 — Cauchyho–Schwarzova nerovnost)</span></p>

*Pro každé $x, y \in V$ platí $|\langle x, y \rangle| \le \|x\| \cdot \|y\|$.*

*Důkaz.* (Reálná verze) Nejprve ukážeme reálnou verzi. Pro $y = o$ platí nerovnost triviálně, tak předpokládejme $y \neq o$. Uvažujme reálnou funkci $f(t) = \langle x + ty, x + ty \rangle \ge 0$ proměnné $t \in \mathbb{R}$. Pak

$$f(t) = \langle x, x \rangle + 2t\langle x, y \rangle + t^2 \langle y, y \rangle.$$

Jedná se o kvadratickou funkci, která je všude nezáporná, nemůže mít tedy dva různé kořeny. Proto je příslušný diskriminant nekladný:

$$4\langle x, y \rangle^2 - 4\langle x, x \rangle \langle y, y \rangle \le 0.$$

Z toho dostáváme $\langle x, y \rangle^2 \le \langle x, x \rangle \langle y, y \rangle$, odmocněním $|\langle x, y \rangle| \le \|x\| \cdot \|y\|$.

*Důkaz.* (Komplexní verze) Pro $y = o$ platí tvrzení triviálně. Buď $y \neq o$ a bez újmy na obecnost předpokládejme, že $\|y\| = 1$. Definujme skalár $\alpha := \langle x, y \rangle$, vektor $z := x - \alpha y$ a upravme

$$0 \le \langle z, z \rangle = \langle x - \alpha y, \, x - \alpha y \rangle = \langle x, x \rangle - \overline{\alpha}\langle x, y \rangle - \alpha \langle y, x \rangle + \alpha \overline{\alpha} \langle y, y \rangle.$$

Protože $\alpha \overline{\alpha} = |\alpha|^2$, $\langle y, y \rangle = 1$ a $\alpha = \langle x, y \rangle$, dostáváme

$$0 \le \langle x, x \rangle - |\alpha|^2 = \langle x, x \rangle - |\langle x, y \rangle|^2.$$

Tudíž $|\alpha|^2 \le \langle x, x \rangle$ a odmocněním obou stran máme $|\langle x, y \rangle| \le \|x\|$.

</div>

Občas se Cauchyho–Schwarzova nerovnost uvádí v ekvivalentní podobě

$$|\langle x, y \rangle|^2 \le \langle x, x \rangle \langle y, y \rangle.$$

Cauchyho–Schwarzova nerovnost se používá pro odvozování dalších výsledků na obecné bázi, nebo i pro konkrétní algebraické výrazy. Například pro standardní skalární součin v $\mathbb{R}^n$ dostaneme nerovnost

$$\left(\sum_{i=1}^{n} x_i y_i\right)^2 \le \left(\sum_{i=1}^{n} x_i^2\right) \left(\sum_{i=1}^{n} y_i^2\right).$$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(Důsledek 8.10 — Trojúhelníková nerovnost)</span></p>

*Pro každé $x, y \in V$ platí $\|x + y\| \le \|x\| + \|y\|$.*

*Důkaz.* Nejprve připomeňme, že pro každé komplexní číslo $z = a + bi$ platí: $z + \overline{z} = 2a = 2\operatorname{Re}(z)$, a dále $a \le |z|$. Nyní můžeme odvodit:

$$\|x + y\|^2 = \langle x + y, x + y \rangle = \langle x, x \rangle + \langle y, y \rangle + 2\operatorname{Re}(\langle x, y \rangle) \le \|x\|^2 + \|y\|^2 + 2|\langle x, y \rangle| \le (\|x\| + \|y\|)^2,$$

kde poslední nerovnost plyne z Cauchyho–Schwarzovy nerovnosti.

</div>

#### Norma obecně

Norma indukovaná skalárním součinem je jen jedním typem normy, pojem normy je ale definován obecněji. My budeme vesměs pracovat s normou indukovanou skalárním součinem, takže následující oddíl je pouze malou odbočkou.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Definice 8.11 — Norma)</span></p>

Buď $V$ vektorový prostor nad $\mathbb{R}$ nebo $\mathbb{C}$. Pak *norma* je zobrazení $\|\cdot\| \colon V \to \mathbb{R}$, splňující:

1. $\|x\| \ge 0$ pro všechna $x \in V$, a rovnost nastane pouze pro $x = 0$,
2. $\|\alpha x\| = |\alpha| \cdot \|x\|$ pro všechna $x \in V$, a pro všechna $\alpha \in \mathbb{R}$ resp. $\alpha \in \mathbb{C}$,
3. $\|x + y\| \le \|x\| + \|y\|$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Tvrzení 8.12)</span></p>

*Norma indukovaná skalárním součinem je normou.*

*Důkaz.* Vlastnost (1) je splněna díky definici normy indukované skalárním součinem. Vlastnost (3) je ukázána v důsledku 8.10. Zbývá vlastnost (2):

$$\|\alpha x\| = \sqrt{\langle \alpha x, \alpha x \rangle} = \sqrt{\alpha \overline{\alpha} \langle x, x \rangle} = \sqrt{\alpha \overline{\alpha}} \sqrt{\langle x, x \rangle} = |\alpha| \cdot \|x\|.$$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 8.13 — Příklady norem v $\mathbb{R}^n$)</span></p>

Speciální třída norem jsou tzv. $p$-normy. Pro $p = 1, 2, \ldots$ definujeme $p$-normu vektoru $x \in \mathbb{R}^n$ jako

$$\|x\|_p = \left(\sum_{i=1}^{n} |x_i|^p\right)^{1/p}.$$

Speciální volby $p$ vedou ke známým normám:

- pro $p = 2$: eukleidovská norma $\|x\|_2 = \sqrt{\sum_{i=1}^{n} x_i^2}$, což je norma indukovaná standardním skalárním součinem,
- pro $p = 1$: součtová norma $\|x\|_1 = \sum_{i=1}^{n} |x_i|$; nazývá se manhattanská norma, protože odpovídá reálným vzdálenostem při procházení pravoúhlé sítě ulic v městě,
- pro $p = \infty$ (limitním přechodem): maximová (Čebyševova) norma $\|x\|\_\infty = \max_{i=1,\ldots,n} |x_i|$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 8.14 — Jednotková koule)</span></p>

Jednotková koule je množina vektorů, které mají normu nanejvýš 1 a tedy jsou od počátku vzdáleny maximálně 1. Formálně definujeme jednotkovou kouli jako

$$\lbrace x \in V \,;\, \|x\| \le 1 \rbrace.$$

Jiné normy mají za jednotkovou kouli jiný geometrický objekt. Každá jednotková koule v $\mathbb{R}^n$ ale musí být uzavřená, omezená, symetrická dle počátku, konvexní (tj., s každými dvěma body obsahuje i jejich spojnici) a počátek leží v jejím vnitřku. Platí i opačné tvrzení — každá množina splňující tyto vlastnosti představuje jednotkovou kouli nějaké normy.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 8.15 — Příklady norem v $\mathcal{C}_{[a,b]}$)</span></p>

Normu spojité funkce $f \colon [a, b] \to \mathbb{R}$ lze zavést analogicky jako pro eukleidovský prostor:

- analogie eukleidovské normy: $\|f\|_2 = \sqrt{\int_a^b f(x)^2 \, dx}$,
- analogie součtové normy: $\|f\|_1 = \int_a^b |f(x)| \, dx$,
- analogie maximové normy: $\|f\|\_\infty = \max_{x \in [a,b]} |f(x)|$,
- analogie $p$-normy: $\|f\|_p = \left(\int_a^b |f(x)|^p \, dx\right)^{1/p}$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Poznámka 8.16 — Rovnoběžníkové pravidlo)</span></p>

Pro normu indukovanou skalárním součinem platí tzv. *rovnoběžníkové pravidlo*:

$$\|x - y\|^2 + \|x + y\|^2 = 2\|x\|^2 + 2\|y\|^2.$$

*Důkaz.* $\|x - y\|^2 + \|x + y\|^2 = \langle x - y, x - y \rangle + \langle x + y, x + y \rangle = 2\langle x, x \rangle + 2\langle y, y \rangle = 2\|x\|^2 + 2\|y\|^2$.

Díky tomu snadno nahlédneme, že součtová a maximová norma nejsou indukovány žádným skalárním součinem. Platí dokonce silnější tvrzení: pokud pro normu platí rovnoběžníkové pravidlo, pak je indukována nějakým skalárním součinem; viz Horn and Johnson [1985].

</div>

Norma umožňuje zavést vzdálenost (neboli metriku) mezi vektory $x, y$ jako $\|x - y\|$. A pokud máme vzdálenost, můžeme zavést limity, etc. Navíc k definici metriky nepotřebujeme ani vektorový prostor, stačí libovolná množina.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Poznámka 8.17 — Metrika)</span></p>

Metriku na množině $M$ definujeme jako je zobrazení $d \colon M^2 \to \mathbb{R}$, splňující:

1. $d(x, y) \ge 0$ pro všechna $x, y \in M$, a rovnost nastane pouze pro $x = y$,
2. $d(x, y) = d(y, x)$ pro všechna $x, y \in M$,
3. $d(x, z) \le d(x, y) + d(y, z)$ pro všechna $x, y, z \in M$.

Každá norma určuje metriku předpisem $d(x, y) := \|x - y\|$, tedy vzdálenost vektorů $x, y$ zavádí jako velikost jejich rozdílu. Opačným směrem to ale obecně neplatí. Existují prostory s metrikou, která není indukována žádnou normou, např. diskrétní metrika $d(x, y) := \lceil \|x - y\|_2 \rceil$, nebo diskrétní metrika $d(x, y) := 1$ pro $x \neq y$ a $d(x, y) := 0$ pro $x = y$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 8.18 — Klasifikace psaných číslic)</span></p>

Ukážeme použití vzdálenosti na vytvoření jednoduchého klasifikátoru pro automatickou identifikaci ručně psaných číslic. Předpokládáme, že každá číslice je zadaná jako obrázek, reprezentovaný maticí $A \in \mathbb{R}^{m \times n}$, tedy pixel obrázku na pozici $(i, j)$ má barvu s číslem $a_{ij}$. Jako vzory použijeme zprůměrované hodnoty z databáze MNIST.

Na prostoru matic proto musíme zavést metriku. K tomuto účelu adaptujeme klasickou eukleidovskou vzdálenost a vzdálenost matic $A, B \in \mathbb{R}^{m \times n}$ definujeme jako

$$\|A - B\| := \sqrt{\sum_{i=1}^{m} \sum_{j=1}^{n} (a_{ij} - b_{ij})^2}.$$

Pokud spočítáme vzdálenosti mezi maticí reprezentující klasifikovaný obrázek a jednotlivými vzory, pak klasifikujeme podle nejmenší vzdálenosti. Takovýto jednoduchý klasifikátor se ale snadno může splést. Náš klasifikátor nedokáže rozpoznat tvary čar (znaménko křivosti atp.), a tudíž je malá vzdálenost je i třeba k obrázku číslo 3.

</div>

### 8.2 Ortonormální báze, Gramova–Schmidtova ortogonalizace

Každý vektorový prostor má bázi. U prostoru se skalárním součinem je přirozené se ptát, zda existuje báze složená z navzájem kolmých vektorů. V této sekci ukážeme, že je to pravda, že taková báze má řadu pozoruhodných vlastností a také odvodíme algoritmus na její nalezení.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Definice 8.19 — Ortogonální a ortonormální systém)</span></p>

Systém vektorů $z_1, \ldots, z_n$ je *ortogonální*, pokud $\langle z_i, z_j \rangle = 0$ pro všechna $i \neq j$. Systém vektorů $z_1, \ldots, z_n$ je *ortonormální*, pokud je ortogonální a $\|z_i\| = 1$ pro všechna $i = 1, \ldots, n$.

</div>

Je-li systém $z_1, \ldots, z_n$ ortonormální, pak je také ortogonální. Naopak to obecně neplatí, ale není problém ortogonální systém zortonormalizovat. Jsou-li $z_1, \ldots, z_n$ nenulové a ortogonální, pak $\frac{1}{\|z_1\|} z_1, \ldots, \frac{1}{\|z_n\|} z_n$ je ortonormální.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 8.20)</span></p>

V prostoru $\mathbb{R}^n$ se standardním skalárním součinem je ortonormálním systémem například kanonická báze $e_1, \ldots, e_n$. Speciálně v rovině $\mathbb{R}^2$ tvoří ortonormální bázi vektory $(1, 0)^T$, $(0, 1)^T$. Jiný příklad ortonormální báze v $\mathbb{R}^2$ je například: $\frac{\sqrt{2}}{2}(1, 1)^T$, $\frac{\sqrt{2}}{2}(-1, 1)^T$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Tvrzení 8.21)</span></p>

*Je-li systém vektorů $z_1, \ldots, z_n$ ortonormální, pak je lineárně nezávislý.*

*Důkaz.* Uvažujme lineární kombinaci $\sum_{i=1}^{n} \alpha_i z_i = o$. Pak pro každé $k = 1, \ldots, n$ platí:

$$0 = \langle o, z_k \rangle = \left\langle \sum_{i=1}^{n} \alpha_i z_i, z_k \right\rangle = \sum_{i=1}^{n} \alpha_i \langle z_i, z_k \rangle = \alpha_k \langle z_k, z_k \rangle = \alpha_k.$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Věta 8.22 — Fourierovy koeficienty)</span></p>

*Buď $z_1, \ldots, z_n$ ortonormální báze prostoru $V$. Pak pro každé $x \in V$ platí $x = \sum_{i=1}^{n} \langle x, z_i \rangle z_i$.*

*Důkaz.* Víme, že $x = \sum_{i=1}^{n} \alpha_i z_i$ a souřadnice $\alpha_1, \ldots, \alpha_n$ jsou jednoznačné (věta 5.33). Nyní pro každé $k = 1, \ldots, n$ platí:

$$\langle x, z_k \rangle = \left\langle \sum_{i=1}^{n} \alpha_i z_i, z_k \right\rangle = \sum_{i=1}^{n} \alpha_i \langle z_i, z_k \rangle = \alpha_k \langle z_k, z_k \rangle = \alpha_k.$$

</div>

Vyjádření $x = \sum_{i=1}^{n} \langle x, z_i \rangle z_i$ se nazývá *Fourierův rozvoj*, a skaláry $\langle x, z_i \rangle$, $i = 1, \ldots, n$ se nazývají *Fourierovy koeficienty*. Geometrický význam Fourierova koeficientu $\langle x, z_i \rangle$ je ten, že $\langle x, z_i \rangle z_i$ udává projekci vektoru $x$ na přímku $\operatorname{span}\lbrace z_i \rbrace$. Potom vektor $x$ lze složit z těchto dílčích projekcí jednoduchým součtem $x = \sum_{i=1}^{n} \langle x, z_i \rangle z_i$ (více o projekcích budeme hovořit v sekci 8.3). Pokud by báze $z_1, \ldots, z_n$ nebyla ortonormální, pak by tato vlastnost už obecně neplatila.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Algoritmus 8.23 — Gramova–Schmidtova ortogonalizace)</span></p>

Vstup: lineárně nezávislé vektory $x_1, \ldots, x_n \in V$.

1. **for** $k := 1$ **to** $n$ **do**
2. $\qquad y_k := x_k - \sum_{j=1}^{k-1} \langle x_k, z_j \rangle z_j$, $\quad$ // vypočítáme kolmici
3. $\qquad z_k := \frac{1}{\|y_k\|} y_k$, $\quad$ // normalizujeme délku na 1
4. **end for**

Výstup: $z_1, \ldots, z_n$ ortonormální báze prostoru $\operatorname{span}\lbrace x_1, \ldots, x_n \rbrace$.

*Důkaz.* (Správnost Gramovy–Schmidtovy ortogonalizace.) Matematickou indukcí podle $n$ dokážeme, že $z_1, \ldots, z_n$ je ortonormální báze prostoru $\operatorname{span}\lbrace x_1, \ldots, x_n \rbrace$. Pro $n = 1$ je $y_1 = x_1 \neq o$, vektor $z_1 = \frac{1}{\|x_1\|} x_1$ je dobře definovaný a $\operatorname{span}\lbrace x_1 \rbrace = \operatorname{span}\lbrace z_1 \rbrace$.

Indukční krok $n \leftarrow n - 1$. Předpokládejme, že $z_1, \ldots, z_{n-1}$ je ortonormální báze prostoru $\operatorname{span}\lbrace x_1, \ldots, x_{n-1} \rbrace$. Kdyby bylo $y_n = o$, tak $x_n = \sum_{j=1}^{n-1} \langle x_n, z_j \rangle z_j$ a $x_n \in \operatorname{span}\lbrace z_1, \ldots, z_{n-1} \rbrace = \operatorname{span}\lbrace x_1, \ldots, x_{n-1} \rbrace$, což by byl spor s lineární nezávislostí vektorů $x_1, \ldots, x_n$. Proto $y_n \neq o$ a $z_n = \frac{1}{\|y_n\|} y_n$ je dobře definovaný a má jednotkovou normu.

Nyní dokážeme, že $z_1, \ldots, z_n$ je ortonormální systém. Z indukčního předpokladu je $z_1, \ldots, z_{n-1}$ ortonormální systém a proto $\langle z_i, z_j \rangle$ je rovno 0 pro $i \neq j$ a rovno 1 pro $i = j$. Stačí ukázat, že $z_n$ je kolmé na ostatní $z_i$ pro $i < n$:

$$\langle z_n, z_i \rangle = \frac{1}{\|y_n\|} \langle y_n, z_i \rangle = \frac{1}{\|y_n\|} \langle x_n, z_i \rangle - \frac{1}{\|y_n\|} \sum_{j=1}^{n-1} \langle x_n, z_j \rangle \langle z_j, z_i \rangle = \frac{1}{\|y_n\|} \langle x_n, z_i \rangle - \frac{1}{\|y_n\|} \langle x_n, z_i \rangle = 0.$$

Zbývá ověřit $\operatorname{span}\lbrace z_1, \ldots, z_n \rbrace = \operatorname{span}\lbrace x_1, \ldots, x_n \rbrace$. Z algoritmu je vidět, že $z_n \in \operatorname{span}\lbrace z_1, \ldots, z_{n-1}, x_n \rbrace \subseteq \operatorname{span}\lbrace x_1, \ldots, x_n \rbrace$, a tedy $\operatorname{span}\lbrace z_1, \ldots, z_n \rbrace \subseteq \operatorname{span}\lbrace x_1, \ldots, x_n \rbrace$. Protože oba prostory mají stejnou dimenzi, nastane rovnost (věta 5.50).

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 8.24 — Gramova–Schmidtova ortogonalizace)</span></p>

Při standardním skalárním součinu chceme najít ortonormální bázi prostoru generovaného vektory

$$x_1 = (1, 0, 1, 0)^T, \quad x_2 = (1, 1, 1, 1)^T, \quad x_3 = (1, 0, 0, 1)^T.$$

Postupujeme přesně podle algoritmu:

$$y_1 := x_1, \qquad z_1 := \frac{1}{\|y_1\|} y_1 = \frac{\sqrt{2}}{2}(1, 0, 1, 0)^T,$$

$$y_2 := x_2 - \langle x_2, z_1 \rangle z_1 = (1, 1, 1, 1)^T - \sqrt{2} \frac{\sqrt{2}}{2}(1, 0, 1, 0)^T = (0, 1, 0, 1)^T,$$

$$z_2 := \frac{1}{\|y_2\|} y_2 = \frac{\sqrt{2}}{2}(0, 1, 0, 1)^T,$$

$$y_3 := x_3 - \langle x_3, z_1 \rangle z_1 - \langle x_3, z_2 \rangle z_2 = (1, 0, 0, 1)^T - \frac{\sqrt{2}\sqrt{2}}{2}(1, 0, 1, 0)^T - \frac{\sqrt{2}\sqrt{2}}{2}(0, 1, 0, 1)^T = \frac{1}{2}(1, -1, -1, 1)^T,$$

$$z_3 := \frac{1}{\|y_3\|} y_3 = \frac{1}{2}(1, -1, -1, 1)^T.$$

Výsledná ortonormální báze se skládá z vektorů $z_1, z_2, z_3$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Poznámka 8.25 — Výpočetní složitost)</span></p>

Pro analýzu výpočetní složitosti algoritmu 8.23 uvažujme vektory $x_1, \ldots, x_n \in \mathbb{R}^m$. Skalární součin dvou vektorů z prostoru $\mathbb{R}^m$ vyžaduje řádově $2m$ aritmetických operací. Krok 2 má tudíž asymptotickou složitost $4m(k-1)$ operací a v kroku 3 je to $3m$. V součtu dostaneme

$$\sum_{k=1}^{n} (4mk - m) = 4m \frac{1}{2} n(n+1) - mn,$$

což řádově dává výpočetní složitost $2mn^2$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(Důsledek 8.26 — Existence ortonormální báze)</span></p>

*Každý konečně generovaný prostor (se skalárním součinem) má ortonormální bázi.*

*Důkaz.* Víme (věta 5.41), že každý konečně generovaný prostor má bázi, a tu můžeme Gramovou–Schmidtovou metodou zortogonalizovat.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(Důsledek 8.27 — Rozšíření ortonormálního systému na ortonormální bázi)</span></p>

*Každý ortonormální systém vektorů v konečně generovaném prostoru lze rozšířit na ortonormální bázi.*

*Důkaz.* Víme (věta 5.49), že každý ortonormální systém vektorů $z_1, \ldots, z_m$ lze rozšířit na bázi $z_1, \ldots, z_m, x_{m+1}, \ldots, x_n$, a tu můžeme Gramovou–Schmidtovou metodou zortogonalizovat na $z_1, \ldots, z_m, z_{m+1}, \ldots, z_n$. Ortogonalizací se totiž prvních $m$ vektorů nezmění.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Věta 8.28 — Besselova nerovnost a Parsevalova rovnost)</span></p>

*Buď $z_1, \ldots, z_n$ ortonormální systém ve $V$ a buď $x \in V$. Pak platí:*

1. *Besselova nerovnost:* $\|x\|^2 \ge \sum_{j=1}^{n} |\langle x, z_j \rangle|^2$,
2. *Parsevalova rovnost:* $\|x\|^2 = \sum_{j=1}^{n} |\langle x, z_j \rangle|^2$ *právě tehdy, když $x \in \operatorname{span}\lbrace z_1, \ldots, z_n \rbrace$.*

*Důkaz.*

(1) Vyplývá z úpravy

$$0 \le \left\langle x - \sum_{j=1}^{n} \langle x, z_j \rangle z_j, \, x - \sum_{j=1}^{n} \langle x, z_j \rangle z_j \right\rangle = \|x\|^2 - \sum_{j=1}^{n} |\langle x, z_j \rangle|^2.$$

(2) Vyplývá z předchozího, neboť rovnost nastane právě tehdy, když $x = \sum_{j=1}^{n} \langle x, z_j \rangle z_j$.

</div>

Besselova nerovnost říká, že norma vektoru $x$ nemůže být nikdy menší než norma jeho projekce do libovolného podprostoru, zde vyjádřeného jako $\operatorname{span}\lbrace z_1, \ldots, z_n \rbrace$.

Parsevalova rovnost ukazuje, že pro vektory blízké počátku musí i jejich souřadnice být dostatečně malé. Dále, rovnost se dá zobecnit i pro nekonečně-dimenzionální prostory jako je $\mathcal{C}\_{[-\pi, \pi]}$, což mj. znamená, že Fourierovy koeficienty v nekonečném rozvoji musí konvergovat k nule.

Parsevalova rovnost také jinými slovy říká, že v jakémkoli konečně generovaném prostoru $V$ se norma libovolného $x \in V$ dá vyjádřit jako standardní eukleidovská norma jeho vektoru souřadnic:

$$\|x\| = \|[x]_B\|_2 = \sqrt{[x]_B^T [x]_B},$$

kde $B$ je ortonormální báze $V$.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Tvrzení 8.29)</span></p>

*Buď $B$ ortonormální báze prostoru $V$ a buď $x, y \in V$. Pak $\langle x, y \rangle = [x]_B^T \overline{[y]_B}$.*

*Důkaz.* Buď $B = \lbrace z_1, \ldots, z_n \rbrace$. Podle věty 8.22 je $[x]_B = (\langle x, z_1 \rangle, \ldots, \langle x, z_n \rangle)^T$. Nyní

$$\langle x, y \rangle = \left\langle \sum_{j=1}^{n} \langle x, z_j \rangle z_j, y \right\rangle = \sum_{j=1}^{n} \langle x, z_j \rangle \overline{\langle y, z_j \rangle} = [x]_B^T \overline{[y]_B}.$$

</div>

Není těžké nahlédnout, že tato věta platí i naopak. Čili dostáváme, že zobrazení $\langle \cdot, \cdot \rangle$ je skalárním součinem na prostoru $V$ právě tehdy, když se dá vyjádřit jako $\langle x, y \rangle = [x]_B^T \overline{[y]_B}$ pro nějakou (či pro libovolnou) ortonormální bázi $B$. Každý skalární součin je tedy standardním skalárním součinem při pohledu z libovolné ortonormální báze.

### 8.3 Ortogonální doplněk a projekce

V této sekci odvodíme metodu na spočítání vzdálenosti bodu od podprostoru (například bodu od přímky, bodu od roviny, ...) a také na určení toho bodu podprostoru, který je danému bodu nejblíže. To umožní řešit jak ryze geometrické úlohy, tak i úlohy, které zdánlivě s geometrií nesouvisí.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Definice 8.30 — Ortogonální doplněk)</span></p>

Buď $V$ vektorový prostor a $M \subseteq V$. Pak *ortogonální doplněk* množiny $M$ je $M^\perp := \lbrace x \in V \,;\, \langle x, y \rangle = 0 \;\forall y \in M \rbrace$.

</div>

Ortogonální doplněk $M^\perp$ tedy obsahuje takové vektory $x$, které jsou kolmé na všechny vektory z $M$ (někdy zkráceně říkáme, že $x$ je kolmé na $M$). Zřejmě platí $\lbrace o \rbrace^\perp = V$ a $V^\perp = \lbrace o \rbrace$.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 8.31)</span></p>

Ortogonální doplněk k vektoru $(2, 5)^T$ je přímka $\operatorname{span}\lbrace (5, -2)^T \rbrace$. Ortogonální doplněk k celé přímce $\operatorname{span}\lbrace (2, 5)^T \rbrace$ je rovněž přímka $\operatorname{span}\lbrace (5, -2)^T \rbrace$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Tvrzení 8.32 — Vlastnosti ortogonálního doplňku množiny)</span></p>

*Buď $V$ vektorový prostor a $M, N \subseteq V$. Pak*

1. *$M^\perp$ je podprostor $V$,*
2. *je-li $M \subseteq N$ pak $M^\perp \supseteq N^\perp$,*
3. *$M^\perp = \operatorname{span}(M)^\perp$.*

*Důkaz.*

(1) Ověříme vlastnosti podprostoru: $o \in M^\perp$ triviálně. Nyní buďte $x_1, x_2 \in M^\perp$. Pak $\langle x_1, y \rangle = \langle x_2, y \rangle = 0$ $\forall y \in M$, tedy i $\langle x_1 + x_2, y \rangle = \langle x_1, y \rangle + \langle x_2, y \rangle = 0$. Nakonec, buď $x \in M^\perp$. Pak pro každý skalár $\alpha$ je $\langle \alpha x, y \rangle = \alpha \langle x, y \rangle = 0$.

(2) Buď $x \in N^\perp$, tedy $\langle x, y \rangle = 0$ $\forall y \in N$. Tím spíš $\langle x, y \rangle = 0$ $\forall y \in M \subseteq N$, a proto $x \in M^\perp$.

(3) $M \subseteq \operatorname{span}(M)$, tedy dle předchozího je $M^\perp \supseteq \operatorname{span}(M)^\perp$. Důkaz druhé inkluze spočívá v tom, že je-li vektor $x$ kolmý na určité vektory, pak je kolmý na jejich lineární kombinace, a tím pádem na jejich lineární obal.

</div>

Vlastnost (3) říká, že ortogonální doplněk prostoru nebo jeho báze je ten samý. To ulehčí práci pro praktické hledání ortogonálního doplňku, protože stačí ověřit kolmost jen na bázické vektory.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Věta 8.33 — Vlastnosti ortogonálního doplňku podprostoru)</span></p>

*Buď $U$ podprostor vektorového prostoru $V$. Potom platí:*

1. *Je-li $z_1, \ldots, z_m$ ortonormální báze $U$, a je-li $z_1, \ldots, z_m, z_{m+1}, \ldots, z_n$ její rozšíření na ortonormální bázi $V$, pak $z_{m+1}, \ldots, z_n$ je ortonormální báze $U^\perp$.*
2. *$\dim V = \dim U + \dim U^\perp$,*
3. *$V = U + U^\perp$,*
4. *$(U^\perp)^\perp = U$,*
5. *$U \cap U^\perp = \lbrace o \rbrace$.*

*Důkaz.*

(1) Zřejmě $z_{m+1}, \ldots, z_n$ je ortonormální systém v $V$, a tudíž stačí dokázat $\operatorname{span}\lbrace z_{m+1}, \ldots, z_n \rbrace = U^\perp$. Inkluze „$\supseteq$". Každý $x \in V$ má Fourierův rozvoj $x = \sum_{i=1}^{n} \langle x, z_i \rangle z_i$. Je-li $x \in U^\perp$, pak $\langle x, z_i \rangle = 0$, $i = 1, \ldots, m$, a tudíž $x = \sum_{i=m+1}^{n} \langle x, z_i \rangle z_i \in \operatorname{span}\lbrace z_{m+1}, \ldots, z_n \rbrace$. Inkluze „$\subseteq$". Buď $x \in \operatorname{span}\lbrace z_{m+1}, \ldots, z_n \rbrace$, pak $x = \sum_{i=m+1}^{n} \langle x, z_i \rangle z_i = \sum_{i=1}^{n} 0 z_i + \sum_{i=m+1}^{n} \langle x, z_i \rangle z_i$. Z jednoznačnosti souřadnic dostáváme $\langle x, z_i \rangle = 0$, $i = 1, \ldots, m$, a tím $x \in U^\perp$.

(2) Z první vlastnosti máme $\dim V = n$, $\dim U = m$, $\dim U^\perp = n - m$.

(3) Z první vlastnosti máme $x = \sum_{i=1}^{m} \langle x, z_i \rangle z_i + \sum_{i=m+1}^{n} \langle x, z_i \rangle z_i \in U + U^\perp$.

(4) Z první vlastnosti je $z_{m+1}, \ldots, z_n$ ortonormální báze $U^\perp$, tedy $z_1, \ldots, z_m$ je ortonormální báze $(U^\perp)^\perp$.

(5) Z předchozího a podle věty 5.56 o dimenzi spojení a průniku je $\dim(U \cap U^\perp) = \dim V - \dim U - \dim U^\perp = 0$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Definice 8.35 — Ortogonální projekce)</span></p>

Buď $V$ vektorový prostor a $U$ jeho podprostor. Pak *projekcí* vektoru $x \in V$ do podprostoru $U$ rozumíme takový vektor $x_U \in U$, který splňuje

$$\|x - x_U\| = \min_{y \in U} \|x - y\|.$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Věta 8.36 — O ortogonální projekci)</span></p>

*Buď $U$ podprostor vektorového prostoru $V$. Pak pro každé $x \in V$ existuje právě jedna projekce $x_U \in U$ do podprostoru $U$. Navíc, je-li $z_1, \ldots, z_m$ ortonormální báze $U$, pak*

$$x_U = \sum_{i=1}^{m} \langle x, z_i \rangle z_i.$$

*Důkaz.* Buď $z_1, \ldots, z_m, z_{m+1}, \ldots, z_n$ rozšíření na ortonormální bázi $V$. Zadefinujme $x_U := \sum_{i=1}^{m} \langle x, z_i \rangle z_i \in U$ a ukažme, že je to hledaný vektor. Nyní

$$x - x_U = \sum_{i=1}^{n} \langle x, z_i \rangle z_i - \sum_{i=1}^{m} \langle x, z_i \rangle z_i = \sum_{i=m+1}^{n} \langle x, z_i \rangle z_i \in U^\perp.$$

Buď $y \in U$ libovolné. Nyní máme $x - x_U \in U^\perp$ a $x_U - y \in U$. Tudíž $(x - x_U) \perp (x_U - y)$ a můžeme použít Pythagorovu větu, která dává

$$\|x - y\|^2 = \|(x - x_U) + (x_U - y)\|^2 = \|x - x_U\|^2 + \|x_U - y\|^2 \ge \|x - x_U\|^2,$$

neboli $\|x - y\| \ge \|x - x_U\|$, což dokazuje minimalitu. Jednoznačnost: rovnost nastane pouze tehdy, když $\|x_U - y\|^2 = 0$, čili když $x_U = y$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 8.37)</span></p>

Chceme najít projekci $x_U$ vektoru $x = (1, 2, 4, 5)^T$ do podprostoru $U$ generovaného vektory

$$x_1 = (1, 0, 1, 0)^T, \quad x_2 = (1, 1, 1, 1)^T, \quad x_3 = (1, 0, 0, 1)^T$$

a určit vzdálenost $x$ od $U$ při standardním skalárním součinu.

Nejprve najdeme ortonormální bázi podprostoru $U$. To jsme již učinili v příkladu 8.24, a ortonormální bázi tvoří vektory

$$z_1 = \frac{\sqrt{2}}{2}(1, 0, 1, 0)^T, \quad z_2 = \frac{\sqrt{2}}{2}(0, 1, 0, 1)^T, \quad z_3 = \frac{1}{2}(1, -1, -1, 1)^T.$$

Nyní najdeme projekci dle vzorce

$$x_U = \sum_{i=1}^{3} \langle x, z_i \rangle z_i = \frac{1}{2}(5, 7, 5, 7)^T.$$

Hledaná vzdálenost je $\|x - x_U\| = \|\frac{1}{2}(-3, -3, 3, 3)^T\| = 3$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 8.38 — Projekce na přímku)</span></p>

Buď $a \in \mathbb{R}^n$ nenulový vektor a uvažujme projekci vektoru $x \in \mathbb{R}^n$ na přímku se směrnicí $a$, čili projekci do podprostoru $U = \operatorname{span}\lbrace a \rbrace$. Ortonormální báze prostoru $U$ je vektor $z = \frac{1}{\|a\|} a$ a podle vzorce (8.2) má projekce vektoru $x$ tvar

$$x_U = \langle x, z \rangle z = \frac{1}{\|a\|^2} \langle x, a \rangle a = \frac{x^T a}{a^T a} a.$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Poznámka 8.39)</span></p>

Projekci jsme již implicitně použili několikrát ještě dříve, než jsme ji formálně zavedli:

- V důkazu komplexní verze Cauchyho–Schwarzovy nerovnosti (věta 8.9). Vektor $\langle x, y \rangle y$ vyjadřoval projekci vektoru $x$ na přímku $\operatorname{span}\lbrace y \rbrace$ a vektor $z$ představoval rozdíl $x$ a jeho projekce.
- Fourierův rozvoj z věty 8.22 je vlastně rozložení vektoru $x$ na součet projekcí na jednotlivé přímky $\operatorname{span}\lbrace z_i \rbrace$, $i = 1, \ldots, n$.
- Gramova–Schmidtova ortogonalizace v $k$-tém cyklu algoritmu 8.23 konstruuje projekci vektoru $x_k$ do podprostoru $\operatorname{span}\lbrace x_1, \ldots, x_{k-1} \rbrace$. Odečtením projekce od vektoru $x_k$ získáme hledaný nakolmený vektor $y_k$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Poznámka 8.40)</span></p>

Vzhledem k vlastnostem (3) a (5) věty 8.33 se dá prostor $V$ vyjádřit jako direktní součet podprostorů $U$ a $U^\perp$, tedy $V = U \oplus U^\perp$ (viz poznámka 5.58). To mj. znamená, že každý vektor $v \in V$ má jednoznačné vyjádření $v = u + u'$, kde $u \in U$ a $u' \in U^\perp$. Podle věty 8.36 je navíc vektor $u$ projekcí vektoru $v$ do $U$, a vektor $u'$ projekcí $v$ do $U^\perp$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Tvrzení 8.41)</span></p>

*Při značení z věty 8.36, pokud nějaké $y \in U$ splňuje $x - y \in U^\perp$, pak $y = x_U$.*

*Důkaz.* Protože $(x - y) \perp (y - x_U)$, použijeme Pythagorovu větu, která říká

$$\|x - x_U\|^2 = \|x - y\|^2 + \|y - x_U\|^2 \ge \|x - y\|^2.$$

Dostáváme $\|x - x_U\| \ge \|x - y\|$, tedy z vlastnosti a jednoznačnosti projekce musí $y = x_U$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 8.42 — Legendreovy polynomy)</span></p>

Uvažujme prostor polynomů $\mathcal{P}^n$. Pokud si uvědomíme, že $\mathcal{P}^n$ je podprostorem prostoru spojitých funkcí $\mathcal{C}\_{[a,b]}$, tak můžeme na $\mathcal{P}^n$ použít standardní skalární součin prostoru $\mathcal{C}\_{[a,b]}$. Pokud zortogonalizujeme vektory $1, x, x^2, \ldots$ speciálně na $\mathcal{C}\_{[-1,1]}$, pak dostaneme tzv. *Legendreovy polynomy*

$$p_0(x) = 1, \quad p_1(x) = x, \quad p_2(x) = \frac{1}{2}(3x^2 - 1), \quad p_3(x) = \frac{1}{2}(5x^3 - 3x), \quad \ldots$$

Tyto polynomy jsou na sebe kolmé, ale z důvodu určitých aplikací jsou znormovány tak, že $n$-tý polynom má normu $2/(2n+1)$.

Legendreovy polynomy můžeme použít třeba k aproximaci funkce polynomem, srov. metodu v sekci 3.6. Pokud funkci $f$ chceme aproximovat polynomem $n$-tého stupně, tak spočítáme projekci $f$ do podprostoru $\mathcal{P}^n$ v tomto skalárním součinu chceme podle věty 8.36 a za ortonormální bázi $\mathcal{P}^n$ použijeme Legendreovy polynomy. Výsledná projekce má třeba tu vlastnost, že ze všech polynomů stupně $n$ je nejblíže k $f$ v normě indukované daným skalárním součinem, což zhruba odpovídá snaze minimalizovat plochu mezi $f$ a polynomem.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 8.43 — Ortonormální systém v prostoru funkcí)</span></p>

V prostoru $\mathcal{C}\_{[-\pi, \pi]}$ existuje spočetný ortonormální systém $z_1, z_2, \ldots$ sestávající z vektorů

$$\frac{1}{\sqrt{2\pi}}, \quad \frac{1}{\sqrt{\pi}} \cos x, \quad \frac{1}{\sqrt{\pi}} \sin x, \quad \frac{1}{\sqrt{\pi}} \cos 2x, \quad \frac{1}{\sqrt{\pi}} \sin 2x, \quad \frac{1}{\sqrt{\pi}} \cos 3x, \quad \frac{1}{\sqrt{\pi}} \sin 3x, \quad \ldots$$

I když to není báze v pravém slova smyslu, každou funkci $f \in \mathcal{C}\_{[-\pi, \pi]}$ lze vyjádřit jako nekonečnou řadu $f(x) = \sum_{i=1}^{\infty} \langle f, z_i \rangle z_i$.

Vyjádření několika prvních členů $f(x) \approx \sum_{i=1}^{k} \langle f, z_i \rangle z_i$, což je vlastně projekce do prostoru $\operatorname{span}\lbrace z_1, \ldots, z_k \rbrace$ dimenze $k$, dává dobrou aproximaci funkce $f(x)$. Takovéto aproximace se používají hojně v oblasti zpracování signálů (např. zvuku).

Konkrétně, spočítejme Fourierův rozvoj funkce $f(x) = x$ na intervalu $[-\pi, \pi]$

$$x = a_0 + \sum_{k=1}^{\infty} (a_k \sin(kx) + b_k \cos(kx)),$$

kde

$$a_0 = \frac{1}{2\pi} \int_{-\pi}^{\pi} x \, dx = 0, \quad a_k = \frac{1}{\pi} \int_{-\pi}^{\pi} x \sin(kx) \, dx = (-1)^{k+1} \frac{2}{k}, \quad b_k = \frac{1}{\pi} \int_{-\pi}^{\pi} x \cos(kx) \, dx = 0.$$

Tedy $x = \sum_{k=1}^{\infty} (-1)^{k+1} \frac{2}{k} \sin(kx)$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Věta 8.44 — Gramova matice)</span></p>

*Buď $U$ podprostor reálného vektorového prostoru $V$. Nechť $U$ má bázi $B = \lbrace w_1, \ldots, w_m \rbrace$. Označme jako Gramovu matici $G \in \mathbb{R}^{m \times m}$ matici s prvky $G_{ij} = \langle w_i, w_j \rangle$. Pak $G$ je regulární maticí a vektor souřadnic $s = [x_U]_B$ projekce $x_U$ libovolného vektoru $x \in V$ do podprostoru $U$ je řešením soustavy*

$$Gs = (\langle w_1, x \rangle, \ldots, \langle w_m, x \rangle)^T.$$

*Důkaz.* Pro důkaz regularity $G$ buď $s \in \mathbb{R}^m$ řešení soustavy $Gs = o$. Pak $i$-tý řádek soustavy rovnice má tvar $\sum_{j=1}^{m} G_{ij} s_j = \langle w_i, \sum_{j=1}^{m} s_j w_j \rangle = 0$, čili $\sum_{j=1}^{m} s_j w_j \in U^\perp \cap U = \lbrace o \rbrace$. Z lineární nezávislosti $w_1, \ldots, w_m$ nutně $s = o$.

Víme, že $x_U$ existuje a je jednoznačná a lze psát ve tvaru $x_U = \sum_{j=1}^{m} \alpha_j w_j$ pro vhodné skaláry $\alpha_j$. Protože $x - x_U \in U^\perp$, dostáváme speciálně $\langle w_i, x - x_U \rangle = 0$, pro všechna $i = 1, \ldots, m$. Dosazením za $x_U$ získáme $\langle w_i, x - \sum_{j=1}^{m} \alpha_j w_j \rangle = 0$, neboli

$$\sum_{j=1}^{m} \alpha_j \langle w_i, w_j \rangle = \langle w_i, x \rangle, \quad i = 1, \ldots, m.$$

Tedy $s := [x_U]_B = (\alpha_1, \ldots, \alpha_m)^T$ řeší soustavu. Z regularity $G$ pak existuje pouze jediné řešení soustavy a odpovídá dané projekci.

</div>

Gramova matice je regulární právě tehdy, když vektory $w_1, \ldots, w_m$ jsou lineárně nezávislé.

### 8.4 Ortogonální doplněk a projekce v $\mathbb{R}^n$

Z minulé sekce víme, jak počítat ortogonální doplněk a projekci pro libovolný konečně generovaný vektorový prostor se skalárním součinem, a to pomocí ortonormální báze. Nyní ukážeme, že v $\mathbb{R}^n$ pro standardní skalární součin lze tyto transformace vyjádřit explicitně a přímo bez počítání ortonormální báze.

#### Ortogonální doplněk

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Věta 8.45 — Ortogonální doplněk v $\mathbb{R}^n$)</span></p>

*Buď $A \in \mathbb{R}^{m \times n}$. Pak $\mathcal{R}(A)^\perp = \operatorname{Ker}(A)$.*

*Důkaz.* Z vlastností ortogonálního doplňku (tvrzení 8.32(3)) víme $\mathcal{R}(A)^\perp = \lbrace A_{1*}, \ldots, A_{m*} \rbrace^\perp$. Tedy $x \in \mathcal{R}(A)^\perp$ právě tehdy, když $x$ je kolmé na řádky matice $A$, neboli $A_{i*} x = 0$ pro všechna $i = 1, \ldots, m$. Ekvivalentně, $Ax = o$, to jest $x \in \operatorname{Ker}(A)$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 8.46)</span></p>

Buď $V$ prostor generovaný vektory $(1, 2, 3)^T$ a $(1, -1, 0)^T$. Chceme-li určit $V^\perp$, tak sestavíme matici

$$A = \begin{pmatrix} 1 & 2 & 3 \\ 1 & -1 & 0 \end{pmatrix},$$

protože $V = \mathcal{R}(A)$. Nyní již stačí nalézt bázi $V^\perp = \operatorname{Ker}(A)$, kterou tvoří například vektor $(1, 1, -1)^T$.

</div>

Charakterizace ortogonálního doplňku má i teoretické důsledky, například vztah matice $A$ a matice $A^T A$. Pozor, pro sloupcové prostory analogie neplatí!

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(Důsledek 8.47)</span></p>

*Buď $A \in \mathbb{R}^{m \times n}$. Pak*

1. *$\operatorname{Ker}(A^T A) = \operatorname{Ker}(A)$,*
2. *$\mathcal{R}(A^T A) = \mathcal{R}(A)$,*
3. *$\operatorname{rank}(A^T A) = \operatorname{rank}(A)$.*

*Důkaz.*

(1) Je-li $x \in \operatorname{Ker}(A)$, pak Ax = o, a tedy také $A^T Ax = A^T o = o$, čímž $x \in \operatorname{Ker}(A^T A)$. Naopak, je-li $x \in \operatorname{Ker}(A^T A)$, pak $A^T Ax = o$. Pronásobením $x^T$ dostaneme $x^T A^T Ax = o$, neboli $\|Ax\|^2 = 0$. Z vlastnosti normy musí $Ax = o$ a tudíž $x \in \operatorname{Ker}(A)$.

(2) $\mathcal{R}(A^T A) = \operatorname{Ker}(A^T A)^\perp = \operatorname{Ker}(A)^\perp = \mathcal{R}(A)$.

(3) Triviálně z předchozího bodu.

</div>

#### Maticové prostory a lineární zobrazení

Pokud lineární zobrazení dané předpisem $f(x) = Ax$, kde $A \in \mathbb{R}^{m \times n}$, je prosté, tak můžeme zavést inverzní zobrazení z prostoru $f(\mathbb{R}^n) = \mathcal{S}(A)$ do prostoru $\mathbb{R}^n$.

Pokud lineární zobrazení $f(x)$ není prosté, tak $\dim f(\mathbb{R}^n) < n$. Jediná možnost, jak zkonstruovat něco jako inverzní zobrazení, je zvolit vhodný podprostor $U$ prostoru $\mathbb{R}^n$ tak, aby $\dim U = \dim f(\mathbb{R}^n)$ a zároveň $f(U) = f(\mathbb{R}^n)$. Potom zobrazení $f(x)$ na omezeném definičním oboru $U$ představuje isomorfismus mezi $U$ a $f(\mathbb{R}^n)$, a tím pádem k němu existuje inverzní zobrazení.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Tvrzení 8.48)</span></p>

*Uvažujme lineární zobrazení $f(x) = Ax$, kde $A \in \mathbb{R}^{m \times n}$. Pokud definiční obor $f(x)$ omezíme pouze na prostor $\mathcal{R}(A)$, tak dostaneme isomorfismus mezi $\mathcal{R}(A)$ a $f(\mathbb{R}^n)$.*

*Důkaz.* Buď $x \in \mathbb{R}^n$. Protože $\mathcal{R}(A)^\perp = \operatorname{Ker}(A)$, lze podle poznámky 8.40 vektor $x$ rozložit jako $x = x^R + x^K$, kde $x^R \in \mathcal{R}(A)$ a $x^K \in \operatorname{Ker}(A)$. Pak

$$f(x) = Ax = A(x^R + x^K) = Ax^R + Ax^K = Ax^R.$$

Každý vektor z $f(\mathbb{R}^n)$ je tudíž obrazem nějakého vektoru z $\mathcal{R}(A)$, neboli $f(\mathcal{R}(A)) = f(\mathbb{R}^n)$. Protože oba prostory $\mathcal{R}(A)$ a $f(\mathbb{R}^n)$ mají stejnou dimenzi (rovnou $\operatorname{rank}(A)$), představuje zobrazení $f(x)$ isomorfismus.

</div>

#### Ortogonální projekce

Nyní odvodíme explicitní vzorec pro projekci vektoru $x$ do podprostoru $U$. Pokud vektory báze podprostoru $U$ dáme do sloupců matice $A$, tak projekci vektoru $x$ do $U$ lze formulovat jako projekci $x$ do $\mathcal{S}(A)$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Věta 8.49 — Ortogonální projekce v $\mathbb{R}^m$)</span></p>

*Buď $A \in \mathbb{R}^{m \times n}$ hodnosti $n$. Pak projekce vektoru $x \in \mathbb{R}^m$ do sloupcového prostoru $\mathcal{S}(A)$ je $x' = A(A^T A)^{-1} A^T x$.*

*Důkaz.* Nejprve si uvědomíme, že $x'$ je dobře definované. Matice $A^T A$ má dimenzi $n$ (důsledek 8.47(3)), tedy je regulární a má inverzi. Podle tvrzení 8.41 stačí nyní ukázat, že $x' \in \mathcal{S}(A)$ a $x - x' \in \mathcal{S}(A)^\perp$. První vlastnost platí, neboť $x' = Az$ pro $z = (A^T A)^{-1} A^T x$. Pro druhou vlastnost stačí ověřit, že $x - x' \in \mathcal{S}(A)^\perp = \mathcal{R}(A^T)^\perp = \operatorname{Ker}(A^T)$, a to plyne z vyjádření

$$A^T(x - x') = A^T(x - A(A^T A)^{-1} A^T x) = A^T x - A^T A(A^T A)^{-1} A^T x = A^T x - A^T x = o.$$

</div>

Poznamenejme, že projekce je lineární zobrazení a podle předchozí věty je $P := A(A^T A)^{-1} A^T$ jeho matice (vzhledem ke kanonické bázi). Navíc tato matice má několik speciálních vlastností:

- Matice $P$ je symetrická.
- Platí $P^2 = P$. Projekce vektoru $x$ je vektor $Px$. Vektor $Px$ již náleží do podprostoru $\mathcal{S}(A)$, a proto jeho projekce je on sám: $P^2 x = Px$.
- Protože $P$ reprezentuje projekci do $\mathcal{S}(A)$, platí $\mathcal{S}(P) = \mathcal{S}(A)$. Hodnost matice $P$ je tedy rovna dimenzi prostoru, do kterého projektujeme, čili $\operatorname{rank}(P) = \operatorname{rank}(A)$. Matice $P$ je tak regulární pouze v případě, kdy $m = n$, tj. $\mathcal{S}(A) = \mathbb{R}^n$.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Tvrzení 8.50)</span></p>

*Matice $P \in \mathbb{R}^{n \times n}$ je maticí projekce právě tehdy, když je symetrická a $P = P^2$.*

*Důkaz.* Jeden směr jsme již nahlédli. Nyní předpokládejme, že $P$ je symetrická a splňuje $P = P^2$, a chceme ukázat, že je maticí projekce na prostor $\mathcal{S}(A)$. Jinými slovy, chceme ukázat, že pro každý vektor $x \in \mathbb{R}^n$ je $Px$ jeho projekce do $\mathcal{S}(A)$. Podle tvrzení 8.41 stačí ukázat, že $x - Px \in \mathcal{S}(A)^\perp$. Tedy $x - Px$ musí být kolmé všechny vektory z $\mathcal{S}(A)$, a ty mají tvar $Py$, kde $y \in \mathbb{R}^n$. To se ale snadno ověří rozepsáním jejich skalárního součinu

$$((I_n - P)x)^T Py = x^T(I_n - P)^T Py = x^T(P - P^2)y = 0.$$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 8.51)</span></p>

Uvažujme problém z příkladu 8.37: spočítání projekce $x_U$ vektoru $x = (1, 2, 4, 5)^T$ do podprostoru $U$ generovaného vektory $x_1 = (1, 0, 1, 0)^T$, $x_2 = (1, 1, 1, 1)^T$, $x_3 = (1, 0, 0, 1)^T$.

Protože pracujeme se standardním skalárním součinem, můžeme projekci spočítat alternativně podle věty 8.49. Sestavíme matici

$$A = \begin{pmatrix} 1 & 1 & 1 \\ 0 & 1 & 0 \\ 1 & 1 & 0 \\ 0 & 1 & 1 \end{pmatrix},$$

jejíž sloupce jsou tvořeny $x_1, x_2, x_3$, a projekce se spočítá dle vzorce

$$x_U = A(A^T A)^{-1} A^T x = \frac{1}{2}(5, 7, 5, 7)^T.$$

Zde navíc

$$P = A(A^T A)^{-1} A^T = \frac{1}{4} \begin{pmatrix} 3 & -1 & 1 & 1 \\ -1 & 3 & 1 & 1 \\ 1 & 1 & 3 & -1 \\ 1 & 1 & -1 & 3 \end{pmatrix}$$

představuje matici projekce jakožto lineárního zobrazení do podprostoru $U$. Takže pokud ji máme takto explicitně vyjádřenou, projekce $x_U$ vektoru $x$ se spočítá snadno jako $x_U = Px$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Poznámka 8.52 — Projekce s ortonormální bází)</span></p>

Označme jako $z_1, \ldots, z_n$ sloupečky matice $A \in \mathbb{R}^{m \times n}$ a nechť tvoří ortonormální systém. Potom $(A^T A)_{ij} = \langle z_i, z_j \rangle$ a tudíž $A^T A = I_n$. Matice projekce $P$ do sloupcového prostoru $\mathcal{S}(A)$ získává jednodušší tvar $P = A(A^T A)^{-1} A^T = AA^T$. Zde si můžeme všimnout paralely s předpisem projekce (8.2), protože projekce vektoru $x \in \mathbb{R}^n$ je

$$Px = AA^T x = \begin{pmatrix} | & | & & | \\ z_1 & z_2 & \cdots & z_n \\ | & | & & | \end{pmatrix} \begin{pmatrix} z_1^T x \\ z_2^T x \\ \vdots \\ z_n^T x \end{pmatrix} = \sum_{i=1}^{n} (z_i^T x) z_i.$$

Rekapitulace: Nechť matice $A$ má ortonormální sloupce. Potom $AA^T$ představuje matici projekce do $\mathcal{S}(A)$ a obecně $AA^T \neq I_m$, ale součin v opačném pořadí dá $A^T A = I_n$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 8.53 — Projekce na přímku podruhé)</span></p>

Speciálně, matice projekce na jednodimenzionální podprostor (přímku) má tvar $P = a(a^T a)^{-1} a^T$, kde $a \in \mathbb{R}^n$ je směrnice přímky. Projekce vektoru $x$ na přímku je pak vektor $Px = a(a^T a)^{-1} a^T x = \frac{a^T x}{a^T a} a$ (srov. příklad 8.38). Pokud navíc směrnici normujeme tak, aby $\|a\|_2 = 1$, potom $a^T a = 1$ a tudíž matice projekce získá jednoduchý tvar $P = aa^T$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Věta 8.54 — Ortogonální projekce do doplňku)</span></p>

*Buď $P \in \mathbb{R}^{n \times n}$ matice projekce do podprostoru $V \subseteq \mathbb{R}^n$. Pak $I - P$ je maticí projekce do $V^\perp$.*

*Důkaz.* Podle věty 8.33 lze každý vektor $x \in \mathbb{R}^n$ jednoznačně rozložit na součet $x = y + z$, kde $y \in V$ a $z \in V^\perp$. Z pohledu věty 8.36 je $y$ projekce $x$ do $V$ a $z$ projekce $x$ do $V^\perp$. Tedy $z = x - y = x - Px = (I - P)x$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 8.55 — Matice projekce do $\operatorname{Ker}(A)$)</span></p>

Věta 8.54 umožňuje elegantně vyjádřit projekci do jádra matice $A \in \mathbb{R}^{m \times n}$. Předpokládejme, že $\operatorname{rank}(A) = m$. Protože $\operatorname{Ker}(A)^\perp = \mathcal{R}(A) = \mathcal{S}(A^T)$, tak matice projekce do $\operatorname{Ker}(A)$ je dána předpisem $I - A^T(AA^T)^{-1} A$, kde $A^T(AA^T)^{-1} A$ je matice projekce do $\mathcal{S}(A^T)$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Poznámka 8.56 — Vzdálenosti podprostorů)</span></p>

Jedním z elegantních využití projekcí v geometrii je řešení vzdáleností afinních podprostorů — vzdálenosti bodu od přímky, vzdálenost bodu od roviny, vzdálenost dvou přímek, vzdálenost bodu od roviny atd. Vzdáleností dvou afinních podprostorů $U + a$, $V + b$ pak rozumíme nejmenší vzdálenost $\|x - y\|$, kde $x \in U + a$, $y \in V + b$. Bez důkazu uvádíme, že nejmenší vzdálenost se vždy nabyde.

Univerzální postup je následující. Mějme $U + a$, $V + b$ dva afinní podprostory prostoru $\mathbb{R}^n$, kde $U = \operatorname{span}\lbrace u_1, \ldots, u_m \rbrace$ a $V = \operatorname{span}\lbrace v_1, \ldots, v_n \rbrace$. Nechť nejmenší vzdálenost se nabyde pro body $x \in U + a$, $y \in V + b$; tyto body jdou vyjádřit jako $x = a + \sum_{i=1}^{m} \alpha_i u_i$, $y = b + \sum_{j=1}^{n} \beta_j v_j$. Vzdálenost těchto dvou bodů je stejná jako vzdálenost bodu $a$ od bodu $b + \sum_{j=1}^{n} \beta_j v_j - \sum_{i=1}^{m} \alpha_i u_i$. Čili hledanou vzdálenost můžeme ekvivalentně vyjádřit jako vzdálenost bodu $a - b$ od afinního podprostoru $U + V$. Posunutím ve směru $-b$ pak vzdálenost spočítáme jako vzdálenost bodu $a - b$ od podprostoru $U + V = \operatorname{span}\lbrace u_1, \ldots, u_m, v_1, \ldots, v_n \rbrace$. To už je standardní úloha, kterou vyřešíme pomocí věty 8.36 resp. věty 8.49 jakožto vzdálenost bodu $a - b$ od své projekce do podprostoru $U + V$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Poznámka 8.57 — Výpočetní složitost)</span></p>

Jaká je složitost výpočtu matice projekce $A(A^T A)^{-1} A^T$ pro $A \in \mathbb{R}^{m \times n}$? Podle poznámek 3.24 a 3.45 stojí výpočet matice $A^T A$ řádově $2mn^2$ operací, její inverze $3n^3$ operací a zbylé dva maticové součiny $2n^2 m + 2nm^2$ operací. Celková asymptotická složitost je pak $3n^3 + 4n^2 m + 2nm^2$.

Pokud nás zajímá pouze projekce vektoru $x \in \mathbb{R}^m$, tak výraz $A(A^T A)^{-1} A^T x$ lze vyhodnotit efektivněji uzávorkováním $A\bigl((A^T A)^{-1} (A^T x)\bigr)$. Spočítání matice $(A^T A)^{-1}$ má opět složitost řádově $2mn^2 + 3n^3$, ale pro součin $A^T x$ dostáváme pouze $2mn$, a pro zbytek $2n^2 + 2nm$. Celkem máme řádově $2mn^2 + 2nm + 3n^3$ operací, což je výrazně méně než pro matici projekce, zejména pokud $m$ je mnohem větší než $n$. Nicméně pokud chceme spočítat pouze projekci jednoho vektoru, je výpočetně ještě trochu výhodnější Gramova–Schmidtova ortogonalizace, která podle poznámky 8.25 stojí asymptoticky pouze $2mn^2$ operací. Samotná projekce při znalosti ortonormální báze pak složitost asymptoticky nezhorší.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Poznámka 8.58 — Projekce a Gramova matice)</span></p>

Předpis pro projekci lze odvodit i z věty 8.44 o Gramově matici. Pokud si jako $w_1, \ldots, w_n$ označíme bázi $\mathcal{S}(A)$ danou ve sloupcích matice $A \in \mathbb{R}^{m \times n}$, tak $\langle w_i, w_j \rangle = (A^T A)_{ij}$. Gramova matice je nyní $A^T A$ a rovnice (8.4) má tvar $A^T As = A^T x$. Z rovnice vyjádříme $s = (A^T A)^{-1} A^T x$, což je vektor souřadnic hledané projekce $x'$. Tudíž

$$x' = \sum_{i=1}^{n} s_i w_i = As = A(A^T A)^{-1} A^T x.$$

</div>

### 8.5 Metoda nejmenších čtverců

Metoda nejmenších čtverců ilustruje další použití věty o projekci. Uvažujme soustavu $Ax = b$, která nemá řešení (typicky, když $m$ je mnohem větší než $n$). V tom případě bychom chtěli nějakou dobrou aproximaci, tj. takový vektor $x$, že levá a pravá strana jsou si co nejblíže. Formálně,

$$\min_{x \in \mathbb{R}^n} \|Ax - b\|.$$

Tento přístup se studuje pro různé normy, ale pro eukleidovskou dostáváme

$$\min_{x \in \mathbb{R}^n} \|Ax - b\|_2^2 = \min_{x \in \mathbb{R}^n} \sum_{j=1}^{n} (A_{j*} x - b_j)^2.$$

Odtud název *metoda nejmenších čtverců*. S využitím věty o projekci najdeme řešení jednoduše. Následující věta říká, že řešení metodou nejmenších čtverců jsou zároveň řešeními soustavy rovnic

$$A^T Ax = A^T b. \tag{8.5}$$

Tato soustava se nazývá *soustava normálních rovnic*. Zajímavé je, že tuto soustavu dostaneme z původní soustavy $Ax = b$ pouhým přenásobením maticí $A^T$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Věta 8.59 — Množina řešení metodou nejmenších čtverců)</span></p>

*Buď $A \in \mathbb{R}^{m \times n}$. Pak množina přibližných řešení soustavy $Ax = b$ metodou nejmenších čtverců je neprázdná a rovna množině řešení normálních rovnic (8.5).*

*Důkaz.* Hledáme vlastně projekci vektoru $b$ do podprostoru $\mathcal{S}(A)$, a tato projekce je vektor tvaru $Ax$, kde $x \in \mathbb{R}^n$. Podle tvrzení 8.41 je $Ax$ projekcí právě tehdy, když $Ax - b \in \mathcal{S}(A)^\perp = \operatorname{Ker}(A^T)$. Jinými slovy, musí platit $A^T(Ax - b) = 0$, neboli $A^T Ax = A^T b$. Tato soustava má řešení, protože projekce musí existovat.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(Důsledek 8.60)</span></p>

*Buď $A \in \mathbb{R}^{m \times n}$ hodnosti $n$. Pak přibližné řešení soustavy $Ax = b$ metodou nejmenších čtverců je $x^* = (A^T A)^{-1} A^T b$, a je jednoznačné.*

</div>

Je-li matice $A$ regulární, pak řešení soustavy $Ax = b$ je $x = A^{-1} b$. Je-li matice $A$ obdélníková s lineárně nezávislými sloupci, pak řešení soustavy $Ax = b$ metodou nejmenších čtverců je $x = (A^T A)^{-1} A^T b$. Na matici $(A^T A)^{-1} A^T$ se můžeme dívat jako na zobecněnou inverzní matici (více viz sekce 13.6). Skutečně, pokud ji vynásobíme maticí $A$ zprava, tak dostaneme $(A^T A)^{-1} A^T A = I_n$. V opačném pořadí tato vlastnost neplatí, čili $(A^T A)^{-1} A^T$ představuje pouze tzv. levou inverzi k $A$.

Metoda nejmenších čtverců má uplatnění v řadě oborů, zejména ve statistice při lineární regresi. Ta studuje chování a odhaduje budoucí vývoj různých veličin, např. globální teploty, HDP, ceny akcií či ropy v čase.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 8.61 — Lineární regrese: vývoj světové populace)</span></p>

Data vývoje světové populace jsou následující:

| rok | 1950 | 1960 | 1970 | 1980 | 1990 | 2000 |
| --- | --- | --- | --- | --- | --- | --- |
| populace (mld.) | 2.519 | 2.982 | 3.692 | 4.435 | 5.263 | 6.070 |

Chceme najít závislost velikosti populace na čase. Předpokládejme, že závislost je lineární. Lineární vztah popíšeme přímkou $y = px + q$, kde $x$ je čas a $y$ velikost populace. Po dosazení dat do rovnic by parametry $p, q$ měly splňovat podmínky

$$2.519 = p \cdot 1950 + q, \quad \ldots, \quad 6.070 = p \cdot 2000 + q.$$

Přesné řešení neexistuje, ale řešení metodou nejmenších čtverců je $p^* = 0.0724$, $q^* = -138.84$.

Výslednou závislost lze využít pro predikce na následující roky. Odhad pro rok 2010 je 6.6943 mld. obyvatel, ve skutečnosti jich bylo 6.853 mld. Ovšem pozor, má smysl vytvářet pouze krátkodobé odhady — v roce 1900 určitě nebyla velikost populace záporná.

</div>

### 8.6 Ortogonální matice

Uvažujme lineární zobrazení v prostoru $\mathbb{R}^n$. Jaké toto zobrazení (potažmo jeho matice) musí být, aby nijak nedeformovalo geometrické objekty? Otočení kolem osy či překlopení podle nadroviny jsou příklady takových zobrazení. Ukážeme, že tato vlastnost souvisí s tzv. ortogonálními maticemi. Ty ale mají dalekosáhlejší význam. Protože mají dobré numerické vlastnosti (viz sekce 1.3 a 3.5), setkáváme se s nimi často v nejrůznějších numerických algoritmech.

V této sekci uvažujeme standardní skalární součin v $\mathbb{R}^n$ resp. $\mathbb{C}^n$ a eukleidovskou normu.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Definice 8.62 — Ortogonální a unitární matice)</span></p>

Matice $Q \in \mathbb{R}^{n \times n}$ je *ortogonální*, pokud $Q^T Q = I_n$. Matice $Q \in \mathbb{C}^{n \times n}$ je *unitární*, pokud $\overline{Q}^T Q = I_n$.

</div>

Pojem unitární matice je zobecnění ortogonálních matic pro komplexní čísla. Nadále ale budeme vesměs pracovat jen s ortogonálními maticemi.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Tvrzení 8.63 — Charakterizace ortogonálních matic)</span></p>

*Buď $Q \in \mathbb{R}^{n \times n}$. Pak následující jsou ekvivalentní:*

1. *$Q$ je ortogonální,*
2. *$Q$ je regulární a $Q^{-1} = Q^T$,*
3. *$QQ^T = I_n$,*
4. *$Q^T$ je ortogonální,*
5. *$Q^{-1}$ existuje a je ortogonální,*
6. *sloupce $Q$ tvoří ortonormální bázi $\mathbb{R}^n$,*
7. *řádky $Q$ tvoří ortonormální bázi $\mathbb{R}^n$.*

*Důkaz.* Stručně. (1)–(5) Je-li $Q$ ortogonální, pak $Q^T Q = I$ a tedy $Q^{-1} = Q^T$; podobně naopak. Dle vlastnosti inverze máme i $QQ^T = I$, neboli $(Q^T)^T Q^T = I$, tedy $Q^T$ je ortogonální. (6): Z rovnosti $Q^T Q = I$ dostáváme porovnáním prvků na pozici $i, j$, že $\langle Q_{*i}, Q_{*j} \rangle = 1$, pokud $i = j$, a $\langle Q_{*i}, Q_{*j} \rangle = 0$, pokud $i \neq j$. Tedy sloupce $Q$ tvoří ortonormální systém. Analogicky naopak.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Tvrzení 8.64 — Součin ortogonálních matic)</span></p>

*Jsou-li $Q_1, Q_2 \in \mathbb{R}^{n \times n}$ ortogonální, pak $Q_1 Q_2$ je ortogonální.*

*Důkaz.* $(Q_1 Q_2)^T Q_1 Q_2 = Q_2^T Q_1^T Q_1 Q_2 = Q_2^T Q_2 = I_n$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 8.65 — Příklady ortogonálních matic)</span></p>

- Jednotková matice $I_n$, nebo k ní opačná $-I_n$.
- *Householderova matice*: $H(a) := I_n - \frac{2}{a^T a} a a^T$, kde $o \neq a \in \mathbb{R}^n$. Její geometrický význam je následující. Nechť $x'$ je projekce bodu $x$ na přímku $\operatorname{span}\lbrace a \rbrace$, a uvažujme lineární zobrazení otočení bodu $x$ dle přímky $\operatorname{span}\lbrace a \rbrace$ o úhel 180°. Pomocí věty 8.49 o projekci dostáváme, že bod $x$ se zobrazí na vektor

  $$x + 2(x' - x) = 2x' - x = 2a(a^T a)^{-1} a^T x - x = \left(2 \frac{aa^T}{a^T a} - I\right) x.$$

  Tedy matice otočení je $\frac{2}{a^T a} aa^T - I_n$. Uvažujme nyní zrcadlení dle nadroviny s normálou $a$. To můžeme reprezentovat jako otočení o 180° dle $a$, a pak překlopení do počátku. Tedy matice tohoto zobrazení je $I_n - \frac{2}{a^T a} aa^T = H(a)$.

  Navíc se dá ukázat, že každou ortogonální matici řádu $n$ lze rozložit jako součin nanejvýš $n$ vhodných Householderových matic. Tudíž lineární zobrazení s ortogonální maticí geometricky reprezentuje složení nanejvýš $n$ zrcadlení.

- *Givensova matice*: Pro $n = 2$ je to matice otočení o úhel $\alpha$ proti směru hodinových ručiček

  $$\begin{pmatrix} \cos \alpha & -\sin \alpha \\ \sin \alpha & \cos \alpha \end{pmatrix}.$$

  Je to tedy matice tvaru $\binom{c\ {-s}}{s\ c}$, kde $c^2 + s^2 = 1$ a každá taková matice odpovídá nějaké matici otočení. Obecně pro dimenzi $n$ je to matice reprezentující otočení o úhel $\alpha$ v rovině os $x_i, x_j$.

  Také z Givensových matic lze složit každou ortogonální matici, ale je jich potřeba v součinu až $\binom{n}{2}$ a případně navíc jedna diagonální matice $s \pm 1$ na diagonále. Geometricky to znamená, že každé lineární zobrazení s ortogonální maticí reprezentuje složení nanejvýš $\binom{n}{2}$ jednoduchých otočení a případně jedno zrcadlení ve směru souřadných os.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Věta 8.66 — Vlastnosti ortogonálních matic)</span></p>

*Buď $Q \in \mathbb{R}^{n \times n}$ ortogonální. Pak:*

1. *$\langle Qx, Qy \rangle = \langle x, y \rangle$ pro každé $x, y \in \mathbb{R}^n$,*
2. *$\|Qx\| = \|x\|$ pro každé $x \in \mathbb{R}^n$,*
3. *$|Q_{ij}| \le 1$ a $|Q_{ij}^{-1}| \le 1$ pro každé $i, j = 1, \ldots, n$,*
4. *$\begin{pmatrix} 1 & o^T \\ o & Q \end{pmatrix}$ je ortogonální matice.*

*Důkaz.*

(1) $\langle Qx, Qy \rangle = (Qx)^T Qy = x^T Q^T Qy = x^T Iy = \langle x, y \rangle$.

(2) $\|Qx\| = \sqrt{\langle Qx, Qx \rangle} = \sqrt{\langle x, x \rangle} = \|x\|$.

(3) Vzhledem k vlastnosti (6) z tvrzení 8.63 je $\|Q_{*j}\| = 1$ pro každé $j = 1, \ldots, n$. Tedy $1 = \|Q_{*j}\|^2 = \sum_{i=1}^{n} q_{ij}^2$, z čehož $q_{ij}^2 \le 1$, a proto $|q_{ij}| \le 1$. Matice $Q^{-1}$ je ortogonální, takže pro ni tvrzení platí také.

(4) Z definice $\begin{pmatrix} 1 & o^T \\ o & Q \end{pmatrix}^T \begin{pmatrix} 1 & o^T \\ o & Q \end{pmatrix} = \begin{pmatrix} 1 & o^T \\ o & Q^T Q \end{pmatrix} = I_{n+1}$.

</div>

Díváme-li se na $Q$ jako na matici příslušného lineárního zobrazení $x \mapsto Qx$, pak vlastnost (1) věty 8.66 říká, že při tom zobrazení se zachovávají úhly, a vlastnost (2) zase říká, že se zachovávají délky. Tvrzení platí i naopak: matice zobrazení zachovávající skalární součin musí být nutně ortogonální (srov. věta 8.68) a dokonce matice zobrazení zachovávající eukleidovskou normu musí být ortogonální. Vlastnost (3) je zase ceněná v numerické matematice, protože $Q$ a $Q^{-1}$ mají omezené velikosti složek. Důležitou vlastností pro numerické počítání je také (2), protože při násobení s ortogonální maticí prvky (a tedy i zaokrouhlovací chyby) nemají tendenci se zvětšovat.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Poznámka 8.67 — Ortogonální matice a Fourierovy koeficienty)</span></p>

Ortogonální matice dávají trochu jiný pohled na Fourierovy koeficienty z věty 8.22. Buď $z_1, \ldots, z_n$ báze prostoru $\mathbb{R}^n$ a buď $v \in \mathbb{R}^n$. Souřadnice vektoru $v$ vzhledem k dané bázi jsou dané vztahem $v = \sum_{i=1}^{n} x_i z_i$. Souřadnice jsou tedy řešením soustavy $Qx = v$, kde sloupce matice $Q$ jsou tvořeny vektory báze, tedy $Q_{*i} = z_i$ pro $i = 1, \ldots, n$. Pokud je báze ortonormální, je matice $Q$ ortogonální a můžeme jednoduše psát

$$x = Q^{-1} v = Q^T v = \begin{pmatrix} z_1^T \\ z_2^T \\ \vdots \\ z_n^T \end{pmatrix} v = \begin{pmatrix} z_1^T v \\ z_2^T v \\ \vdots \\ z_n^T v \end{pmatrix}.$$

Opět tedy dostáváme, že $i$-tá souřadnice $x_i$ vektoru $v$ má hodnotu $\langle z_i, v \rangle = z_i^T v$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Věta 8.68 — Ortogonální matice a lineární zobrazení)</span></p>

*Buďte $U, V$ prostory nad $\mathbb{R}$ s libovolným skalárním součinem a $f \colon U \to V$ lineární zobrazení. Nechť $B_U$ resp. $B_V$ je ortonormální báze $U$ resp. $V$. Pak matice zobrazení ${}_{B_V}[f]\_{B_U}$ je ortogonální právě tehdy, když $\langle f(x), f(y) \rangle = \langle x, y \rangle$ pro každé $x, y \in U$.*

*Důkaz.* Podle tvrzení 8.29 a vlastností matice zobrazení je

$$\langle x, y \rangle = [x]\_{B_U}^T \cdot [y]\_{B_U},$$

$$\langle f(x), f(y) \rangle = [f(x)]\_{B_V}^T \cdot [f(y)]\_{B_V} = \left({}_{B_V}[f]\_{B_U} \cdot [x]\_{B_U}\right)^T \cdot {}_{B_V}[f]\_{B_U} \cdot [y]\_{B_U}.$$

Tudíž, je-li ${}_{B_V}[f]\_{B_U}$ ortogonální, pak rovnost $\langle f(x), f(y) \rangle = \langle x, y \rangle$ platí pro každé $x, y \in U$, neboť souřadnice jsou jednotkové vektory. Naopak, pokud rovnost $\langle f(x), f(y) \rangle = \langle x, y \rangle$ platí pro každé $x, y \in U$, pak dosadíme-li za $x$ a $y$ konkrétně $i$-tý a $j$-tý vektor báze $B_U$, máme $[x]\_{B_U} = e_i$, $[y]\_{B_U} = e_j$, a proto

$$(I_n)_{ij} = e_i^T e_j = [x]\_{B_U}^T [y]\_{B_U} = \langle x, y \rangle = \langle f(x), f(y) \rangle = e_i^T \cdot {}_{B_V}[f]\_{B_U}^T \cdot {}_{B_V}[f]\_{B_U} \cdot e_j = \left({}_{B_V}[f]\_{B_U}^T \cdot {}_{B_V}[f]\_{B_U}\right)\_{ij}.$$

Tímto po složkách dostáváme rovnost $I_n = {}_{B_V}[f]\_{B_U}^T \cdot {}_{B_V}[f]\_{B_U}$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Tvrzení 8.69 — Ortogonální matice a matice přechodu)</span></p>

*Buď $V$ prostor nad $\mathbb{R}$ s libovolným skalárním součinem a $B_1, B_2$ dvě jeho báze. Jakékoli dvě z následujících vlastností implikují tu třetí:*

1. *$B_1$ je ortonormální báze,*
2. *$B_2$ je ortonormální báze,*
3. *${}_{B_2}[id]\_{B_1}$ je ortogonální matice.*

*Důkaz.* Implikace „(1), (2) $\Rightarrow$ (3)". Plyne z věty 8.68, neboť identita zachovává skalární součin. Implikace „(2), (3) $\Rightarrow$ (1)". Buď $B_1 = \lbrace x_1, \ldots, x_n \rbrace$. Z definice jsou sloupce ${}_{B_2}[id]\_{B_1}$ tvořeny vektory $[x_i]\_{B_2}$, které jsou (díky ortogonalitě matice přechodu) ortonormální při standardním skalárním součinu v $\mathbb{R}^n$. Podle tvrzení 8.29 pak $\langle x_i, x_j \rangle = [x_i]\_{B_2}^T [x_j]\_{B_2}$, což je 1 pro $i = j$ a 0 jinak. Implikace „(3), (1) $\Rightarrow$ (2)". Platí z předchozího ze symetrie, neboť ${}_{B_1}[id]\_{B_2} = {}_{B_2}[id]\_{B_1}^{-1}$.

</div>

### Shrnutí ke kapitole 8

Skalární součin zavádí speciální součin dvou vektorů, kdy výsledkem je skalár. Má-li vektorový prostor vybaven skalárním součinem, pak tento skalární součin přirozeně na prostoru definuje také normu, tedy velikost vektoru. A norma pak definuje vzdálenost vektorů jako normu jejich rozdílu. Oba pojmy potřebujeme k tomu, abychom byli schopni měřit v prostoru, ale taky třeba vyjádřit, že posloupnost vektorů konverguje.

Skalární součin dále přirozeně zavádí kolmost vektorů. Ortonormální báze je báze složená z vektorů velikosti 1 a navzájem kolmých. S takovouto bází se pak jednoduše počítají souřadnice, projekce aj. Ortonormální bázi umíme sestrojit Gramovou–Schmidtovou ortogonalizační metodou. Ačkoli jsme pojem skalárního součinu definovali abstraktně, ukázalo se, že každý skalární součin má podobu standardního skalárního součinu v souřadném systému (libovolné) ortonormální báze.

Ortogonální projekce je zobrazení, které vektor zobrazí na jemu nejbližší v daném podprostoru. Přímka, vedená od vektoru k jeho projekci, musí být kolmá na podprostor (odtud „ortogonální" projekce). Projekce se snadno spočítá, pokud známe ortonormální bázi podprostoru. V opačném případě použijeme maticový vzoreček. Projekce je velmi užitečný nástroj nejen v geometrii, kde nám umožňuje elegantně vyjádřit vzdálenosti různých objektů. Jako negeometrickou aplikaci jsme uvedli metodu nejmenších čtverců, která počítá nejlepší přibližné řešení přeurčené soustavy rovnic.

Algebraicky jsou ortogonální matice takové matice, jejichž inverzní matice se jednoduše vyjádří jako transpozice. Ortogonální matice pak geometricky reprezentují lineární zobrazení, které nedeformují objekty — zachovávají úhly i vzdálenosti. Tato zobrazení se dají vždy vyjádřit jako složení konečně mnoha rotací a zrcadlení. Geometrická podstata se odráží i v numerických vlastnostech — počítání s ortogonálními maticemi je výhodné, protože zaokrouhlovací chyby se tolik neamplifikují.

---

## Kapitola 9 — Determinanty

Determinanty byly vyvinuty pro účely řešení čtvercové soustavy lineárních rovnic a dávají explicitní vzorec pro jejich řešení (viz věta 9.15). Za autora determinantu se považuje Gottfried Wilhelm Leibniz a nezávisle na něm jej objevil stejného roku 1683 japonský matematik Seki Kōwa. Samotný pojem „determinant" pochází od Gausse (*Disquisitiones arithmeticae*, 1801). Nicméně ukázalo se, že determinant sám o sobě je jistá charakteristika čtvercové matice s řadou různých uplatnění.

Připomeňme, že $S_n$ značí množinu všech permutací na množině $\lbrace 1, \ldots, n \rbrace$, viz sekce 4.2.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Definice 9.1 — Determinant)</span></p>

Buď $A \in \mathbb{T}^{n \times n}$. Pak *determinant* matice $A$ je číslo

$$\det(A) = \sum_{p \in S_n} \operatorname{sgn}(p) \prod_{i=1}^{n} a_{i, p(i)} = \sum_{p \in S_n} \operatorname{sgn}(p) \, a_{1, p(1)} \ldots a_{n, p(n)}.$$

Značení: $\det(A)$ nebo $|A|$.

</div>

Co vlastně říká vzoreček z definice determinantu? Každý sčítanec má tvar $\operatorname{sgn}(p) \, a_{1,p(1)} \ldots a_{n,p(n)}$, což odpovídá tomu, že v matici $A$ vybereme $n$ prvků tak, že z každého řádku a sloupce máme právě jeden. Tyto prvky pak mezi sebou vynásobíme a ještě sčítanci přiřadíme kladné či záporné znaménko podle toho, jaké bylo znaménko permutace, která tyto prvky určovala.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 9.2 — Determinant matice řádu 2 a 3)</span></p>

Matice řádu 2 má determinant

$$\det \begin{pmatrix} a_{11} & a_{12} \\ a_{21} & a_{22} \end{pmatrix} = a_{11} a_{22} - a_{21} a_{12}.$$

Matice řádu 3 má determinant

$$\det \begin{pmatrix} a_{11} & a_{12} & a_{13} \\ a_{21} & a_{22} & a_{23} \\ a_{31} & a_{32} & a_{33} \end{pmatrix} = a_{11} a_{22} a_{33} + a_{21} a_{32} a_{13} + a_{31} a_{12} a_{23} - a_{31} a_{22} a_{13} - a_{11} a_{32} a_{23} - a_{21} a_{12} a_{33}.$$

</div>

Počítat determinanty z definice pro větší matice je obecně značně neefektivní, protože vyžaduje zpracovat $n!$ sčítanců. Výpočet je jednodušší jen pro speciální matice. Takovou maticí je například horní trojúhelníková matice, tj. matice $A \in \mathbb{T}^{n \times n}$, pro kterou $a_{ij} = 0$ pro $i > j$.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Tvrzení 9.3 — Determinant trojúhelníkové matice)</span></p>

*Buď $A \in \mathbb{T}^{n \times n}$ horní trojúhelníková matice. Pak $\det(A) = a_{1,p(1)} \ldots a_{n,p(n)}$.*

*Důkaz.* Protože matice $A$ je horní trojúhelníková, tak činitel $a_{n,p(n)}$ je nenulový pouze pokud $p(n) = n$. Aby byl činitel $a_{n-1,p(n-1)}$ nenulový, musí buď $p(n-1) = n$ nebo $p(n-1) = n-1$. První možnost je vyloučena vzhledem k $p(n) = n$, tudíž $p(n-1) = n-1$. Opakováním tohoto postupu dospějeme k tomu, že člen (9.1) je nenulový pouze pro permutaci identitu. Proto $\det(A) = a_{1,1} \ldots a_{n,n}$, tj. determinant je roven součinu diagonálních prvků. Jako důsledek pak speciálně $\det(I_n) = 1$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Tvrzení 9.4 — Determinant transpozice)</span></p>

*Buď $A \in \mathbb{T}^{n \times n}$. Pak $\det(A^T) = \det(A)$.*

*Důkaz.*

$$\det(A^T) = \sum_{p \in S_n} \operatorname{sgn}(p) \prod_{i=1}^{n} A^T_{i,p(i)} = \sum_{p \in S_n} \operatorname{sgn}(p) \prod_{i=1}^{n} a_{p(i),i} = \sum_{p \in S_n} \operatorname{sgn}(p^{-1}) \prod_{i=1}^{n} a_{i,p^{-1}(i)} = \sum_{q \in S_n} \operatorname{sgn}(q) \prod_{i=1}^{n} a_{i,q(i)} = \det(A).$$

</div>

Pro determinanty obecně $\det(A + B) \neq \det(A) + \det(B)$, ani není znám jednoduchý vzoreček na determinant součtu matic. Výjimkou je následující speciální případ řádkové linearity.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Věta 9.5 — Řádková linearita determinantu)</span></p>

*Buď $A \in \mathbb{T}^{n \times n}$, $b \in \mathbb{T}^n$. Pak pro libovolné $i = 1, \ldots, n$ platí:*

$$\det(A + e_i b^T) = \det(A) + \det(A + e_i(b^T - A_{i*})).$$

*Jinými slovy,*

$$\det \begin{pmatrix} A_{1*} \\ \vdots \\ a_{i1} + b_1 & \ldots & a_{in} + b_n \\ \vdots \\ A_{n*} \end{pmatrix} = \det \begin{pmatrix} A_{1*} \\ \vdots \\ a_{i1} & \ldots & a_{in} \\ \vdots \\ A_{n*} \end{pmatrix} + \det \begin{pmatrix} A_{1*} \\ \vdots \\ b_1 & \ldots & b_n \\ \vdots \\ A_{n*} \end{pmatrix}.$$

</div>

Vzhledem k tvrzení 9.4 je determinant nejen řádkově, ale i sloupcově lineární.

### 9.1 Determinant a elementární úpravy

Naším plánem je k výpočtu determinantu využít Gaussovu eliminaci. K tomu musíme nejprve umět spočítat determinant matice v odstupňovaném tvaru, a vědět, jak hodnotu determinantu ovlivňují elementární řádkové úpravy. Na první otázku je jednoduchá odpověď, protože matice v odstupňovaném tvaru je zároveň horní trojúhelníková, a tudíž je její determinant roven součinu diagonálních prvků. Druhou otázku zodpovíme rozborem jednotlivých elementárních úprav. Nechť matice $A'$ vznikne z $A$ nějakou elementární úpravou:

1. **Vynásobení $i$-tého řádku číslem $\alpha \in \mathbb{T}$:** $\det(A') = \alpha \det(A)$.

2. **Výměna $i$-tého a $j$-tého řádku:** $\det(A') = -\det(A)$.

3. **Přičtení $\alpha$-násobku $j$-tého řádku k $i$-tému, přičemž $i \neq j$:** $\det(A') = \det(A)$.

*Důkazy.*

(1) Plyne přímo z definice — vytkneme $\alpha$ z $i$-tého řádku.

(2) Označme transpozici $t = (i, j)$. Pak $\det(A') = \sum_{p \in S_n} \operatorname{sgn}(p) a'_{1,p(1)} \ldots a'_{n,p(n)}$, kde $a'_{i,p(i)} = a_{j,p(i)}$, $a'_{j,p(j)} = a_{i,p(j)}$ atd. Substitucí $q = p \circ t$ dostaneme $\det(A') = -\det(A)$.

(3) Z řádkové linearity determinantu a důsledku 9.6 (matice se dvěma stejnými řádky má nulový determinant) dostáváme $\det(A') = \det(A) + \alpha \cdot 0 = \det(A)$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(Důsledek 9.6)</span></p>

*Pokud má matice $A \in \mathbb{T}^{n \times n}$ dva stejné řádky, pak $\det(A) = 0$.*

*Důkaz (pro tělesa charakteristiky $\neq 2$).* Prohozením těchto dvou stejných řádků dostaneme $\det(A) = -\det(A)$, a tedy $\det(A) = 0$.

*Důkaz (pro obecná tělesa).* Pro $\mathbb{Z}_2$ je $1 = -1$, a proto musíme postupovat jinak. Definujme transpozici $t := (i, j)$, kde $i, j$ jsou indexy stejných řádků. Nechť $S'_n$ je množina sudých permutací z $S_n$. Pak $S_n$ lze disjunktně rozložit na $S'_n$ a $\lbrace p \circ t;\, p \in S'_n \rbrace$. Tudíž

$$\det(A) = \sum_{p \in S'_n} \operatorname{sgn}(p) \prod_{i=1}^{n} a_{i,p(i)} + \sum_{p \in S'_n} \operatorname{sgn}(p \circ t) \prod_{i=1}^{n} a_{i,p \circ t(i)} = \sum_{p \in S'_n} \operatorname{sgn}(p) \prod_{i=1}^{n} a_{i,p(i)} - \sum_{p \in S'_n} \operatorname{sgn}(p) \prod_{i=1}^{n} a_{i,p(i)} = 0.$$

</div>

Výše zmíněná pozorování mají několik důsledků: Pro libovolnou matici $A \in \mathbb{T}^{n \times n}$ je $\det(\alpha A) = \alpha^n \det(A)$. Dále, obsahuje-li $A$ nulový řádek nebo sloupec, tak $\det(A) = 0$.

Hlavní význam vlivu elementárních úprav na determinant je, že determinanty můžeme počítat pomocí Gaussovy eliminace:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Algoritmus 9.7 — Výpočet determinantu pomocí REF)</span></p>

Převeď matici $A$ do odstupňovaného tvaru $A'$ a pamatuj si případné změny determinantu v koeficientu $c$; pak $\det(A)$ je roven součinu $c^{-1}$ a diagonálních prvků matice $A'$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 9.8 — Výpočet determinantu pomocí elementárních řádkových úprav)</span></p>

$$|A| = \begin{vmatrix} 1 & 2 & 3 & 4 \\ 1 & 2 & 1 & 3 \\ 2 & 5 & 5 & 5 \\ 0 & 2 & -3 & -4 \end{vmatrix} = \begin{vmatrix} 1 & 2 & 3 & 4 \\ 0 & 0 & -2 & -1 \\ 0 & 1 & -1 & -3 \\ 0 & 2 & -3 & -4 \end{vmatrix} = -\begin{vmatrix} 1 & 2 & 3 & 4 \\ 0 & 1 & -1 & -3 \\ 0 & 0 & -2 & -1 \\ 0 & 0 & -1 & 2 \end{vmatrix}$$

$$= -\begin{vmatrix} 1 & 2 & 3 & 4 \\ 0 & 1 & -1 & -3 \\ 0 & 0 & -2 & -1 \\ 0 & 0 & -1 & 2 \end{vmatrix} = 2 \begin{vmatrix} 1 & 2 & 3 & 4 \\ 0 & 1 & -1 & -3 \\ 0 & 0 & 1 & 0.5 \\ 0 & 0 & -1 & 2 \end{vmatrix} = 2 \begin{vmatrix} 1 & 2 & 3 & 4 \\ 0 & 1 & -1 & -3 \\ 0 & 0 & 1 & 0.5 \\ 0 & 0 & 0 & 2.5 \end{vmatrix} = 5.$$

</div>

### 9.2 Další vlastnosti determinantu

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Věta 9.9 — Kriterium regularity)</span></p>

*Matice $A \in \mathbb{T}^{n \times n}$ je regulární právě tehdy, když $\det(A) \neq 0$.*

*Důkaz.* Převedeme matici $A$ elementárními úpravami na odstupňovaný tvar $A'$, ty mohou měnit hodnotu determinantu, ale nikoli jeho (ne)nulovost. Pak $A$ je regulární právě tehdy, když $A'$ má na diagonále nenulová čísla.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Poznámka 9.10 — Míra regularity)</span></p>

Věta 9.9 umožňuje zavést jakousi míru regularity. Čím je $\det(A)$ blíže k 0, tím je matice $A$ blíž k nějaké singulární matici. Příkladem je Hilbertova matice $H_n$ (viz příklad 3.51), která je špatně podmíněná, protože je „skoro" singulární. Skutečně, jak ukazuje tabulka, determinant matice je velmi blízko nule.

| $n$ | $\det(H_n)$ |
| --- | --- |
| 4 | $\approx 10^{-7}$ |
| 6 | $\approx 10^{-18}$ |
| 8 | $\approx 10^{-33}$ |
| 10 | $\approx 10^{-53}$ |

Tato míra není ale ideální (lepší je např. pomocí vlastních nebo singulárních čísel, viz sekce 13.5), protože je hodně citlivá ke škálování. Uvažujme například matici $0.1 I_n$, pro níž $\det(0.1 I_n) = 10^{-n}$. Přestože $10^{-n}$ může být libovolně malé číslo, samotná matice má k singularitě relativně daleko.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Věta 9.11 — Multiplikativnost determinantu)</span></p>

*Pro každé $A, B \in \mathbb{T}^{n \times n}$ platí $\det(AB) = \det(A) \det(B)$.*

*Důkaz.* (1) Nejprve uvažme speciální případ, když $A$ je matice elementární úpravy:

1. $A = E_i(\alpha)$, vynásobení $i$-tého řádku číslem $\alpha$. Potom $\det(AB) = \alpha \det(B)$ a $\det(A) \det(B) = \alpha \det(B)$.
2. $A = E_{ij}$, prohození $i$-tého a $j$-tého řádku. Pak $\det(AB) = -\det(B)$ a $\det(A) \det(B) = -1 \det(B)$.
3. $A = E_{ij}(\alpha)$, přičtení $\alpha$-násobku $j$-tého řádku k $i$-tému. Pak $\det(AB) = \det(B)$ a $\det(A) \det(B) = 1 \det(B)$.

Tedy rovnost platí ve všech případech.

(2) Nyní uvažme obecný případ. Je-li $A$ singulární, pak i $AB$ je singulární (tvrzení 3.30) a tedy podle věty 9.9 je $\det(AB) = 0 = \det(A) \det(B)$. Je-li $A$ regulární, pak jde rozložit na součin elementárních matic $A = E_1 \ldots E_k$. Nyní postupujme matematickou indukcí podle $k$. Případ $k = 1$ máme vyřešený v bodě (1). Podle indukčního předpokladu a z bodu (1) dostáváme

$$\det(AB) = \det(E_1(E_2 \ldots E_k B)) = \det(E_1) \det((E_2 \ldots E_k) B) = \det(E_1) \det(E_2 \ldots E_k) \det(B) = \det(A) \det(B).$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(Důsledek 9.12)</span></p>

*Buď $A \in \mathbb{T}^{n \times n}$ regulární, pak $\det(A^{-1}) = \det(A)^{-1}$.*

*Důkaz.* $1 = \det(I_n) = \det(A A^{-1}) = \det(A) \det(A^{-1})$.

</div>

Nyní ukážeme rekurentní vzoreček na výpočet determinantu.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Věta 9.13 — Laplaceův rozvoj podle $i$-tého řádku)</span></p>

*Buď $A \in \mathbb{T}^{n \times n}$, $n \ge 2$. Pak pro každé $i = 1, \ldots, n$ platí*

$$\det(A) = \sum_{j=1}^{n} (-1)^{i+j} a_{ij} \det(A^{ij}),$$

*kde $A^{ij}$ je matice vzniklá z $A$ vyškrtnutím $i$-tého řádku a $j$-tého sloupce.*

*Poznámka.* Podobně jako podle řádku můžeme rozvíjet podle libovolného sloupce.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 9.14 — Laplaceův rozvoj podle 4. řádku)</span></p>

$$\begin{vmatrix} 1 & 2 & 3 & 4 \\ 1 & 2 & 1 & 2 \\ 2 & 5 & 5 & 5 \\ 0 & 2 & -4 & -4 \end{vmatrix} = (-1)^{4+1} \cdot 0 \cdot \begin{vmatrix} 2 & 3 & 4 \\ 2 & 1 & 2 \\ 5 & 5 & 5 \end{vmatrix} + (-1)^{4+2} \cdot 2 \cdot \begin{vmatrix} 1 & 3 & 4 \\ 1 & 1 & 2 \\ 2 & 5 & 5 \end{vmatrix}$$

$$+ (-1)^{4+3} \cdot (-4) \cdot \begin{vmatrix} 1 & 2 & 4 \\ 1 & 2 & 2 \\ 2 & 5 & 5 \end{vmatrix} + (-1)^{4+4} \cdot (-4) \cdot \begin{vmatrix} 1 & 2 & 3 \\ 1 & 2 & 1 \\ 2 & 5 & 5 \end{vmatrix}$$

$$= 0 + 2 \cdot 4 + 4 \cdot 2 - 4 \cdot 2 = 8.$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Věta 9.15 — Cramerovo pravidlo)</span></p>

*Buď $A \in \mathbb{T}^{n \times n}$ regulární, $b \in \mathbb{T}^n$. Pak řešení soustavy $Ax = b$ je dáno vzorcem*

$$x_i = \frac{\det(A + (b - A_{*i}) e_i^T)}{\det(A)}, \quad i = 1, \ldots, n.$$

*Důkaz.* Buď $x$ řešení soustavy $Ax = b$; díky regularitě $A$ řešení existuje a je jednoznačné. Rovnost rozepíšeme $\sum_{j=1}^{n} A_{*j} x_j = b$. Ze sloupcové linearity determinantu dostaneme

$$\det(A + (b - A_{*i}) e_i^T) = \det(A_{*1} | \ldots | b | \ldots | A_{*n}) = \det(A_{*1} | \ldots | \sum_{j=1}^{n} A_{*j} x_j | \ldots | A_{*n})$$

$$= \sum_{j=1}^{n} \det(A_{*1} | \ldots | A_{*j} | \ldots | A_{*n}) x_j = \det(A_{*1} | \ldots | A_{*i} | \ldots | A_{*n}) x_i = \det(A) x_i.$$

Nyní stačí obě strany podělit číslem $\det(A) \neq 0$.

</div>

Cramerovo pravidlo z roku 1750 je pojmenováno po švýcarském matematikovi Gabrielu Cramerovi. Ve své době to byl populární nástroj na řešení soustav lineárních rovnic. Dnes se pro praktické výpočty již nepoužívá, protože výpočet řešení soustavy pomocí $n + 1$ determinantů není příliš efektivní z hlediska výpočetního času. Navíc má horší numerické vlastnosti. Význam determinantu je spíše teoretický, mimo jiné ukazuje a dává:

- Explicitní vyjádření řešení soustavy lineárních rovnic.
- Spojitost řešení vzhledem k prvkům matice $A$ a vektoru $b$. Formálně, zobrazení $(A, b) \mapsto A^{-1} b$ je spojité na definičním oboru regulárních matic $A$.
- Odhad velikosti popisu řešení z velikosti popisu vstupních hodnot.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 9.16 — Cramerovo pravidlo)</span></p>

Řešení soustavy rovnic

$$\left(\begin{array}{ccc|c} 1 & 2 & 3 & 1 \\ 1 & 2 & 1 & 3 \\ 2 & 5 & 5 & 4 \end{array}\right)$$

spočítáme po složkách

$$x_1 = \frac{\begin{vmatrix} 1 & 2 & 3 \\ 3 & 2 & 1 \\ 4 & 5 & 5 \end{vmatrix}}{\begin{vmatrix} 1 & 2 & 3 \\ 1 & 2 & 1 \\ 2 & 5 & 5 \end{vmatrix}} = \frac{4}{2} = 2, \quad x_2 = \frac{\begin{vmatrix} 1 & 1 & 3 \\ 1 & 3 & 1 \\ 2 & 4 & 5 \end{vmatrix}}{\begin{vmatrix} 1 & 2 & 3 \\ 1 & 2 & 1 \\ 2 & 5 & 5 \end{vmatrix}} = \frac{2}{2} = 1, \quad x_3 = \frac{\begin{vmatrix} 1 & 2 & 1 \\ 1 & 2 & 3 \\ 2 & 5 & 4 \end{vmatrix}}{\begin{vmatrix} 1 & 2 & 3 \\ 1 & 2 & 1 \\ 2 & 5 & 5 \end{vmatrix}} = \frac{-2}{2} = -1.$$

Řešením je tedy vektor $x = (2, 1, -1)^T$.

</div>

### 9.3 Adjungovaná matice

Adjungovaná matice úzce souvisí s determinanty a maticovou inverzí. Využijeme ji při odvozování Cayleyho–Hamiltonovy věty (věta 10.20), ale čtenář se s ní může potkat např. v kryptografii nebo při odvozování vzorečku pro derivaci determinantu.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Definice 9.17 — Adjungovaná matice)</span></p>

Buď $A \in \mathbb{T}^{n \times n}$ a $n \ge 2$. Pak *adjungovaná matice* $\operatorname{adj}(A) \in \mathbb{T}^{n \times n}$ má složky

$$\operatorname{adj}(A)_{ij} = (-1)^{i+j} \det(A^{ji}), \quad i, j = 1, \ldots, n,$$

kde $A^{ji}$ opět značí matici vzniklou z $A$ vyškrtnutím $j$-tého řádku a $i$-tého sloupce.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Věta 9.18 — O adjungované matici)</span></p>

*Pro každou matici $A \in \mathbb{T}^{n \times n}$ platí $A \operatorname{adj}(A) = \det(A) I_n$.*

*Důkaz.* Odvodíme

$$(A \operatorname{adj}(A))_{ij} = \sum_{k=1}^{n} A_{ik} \operatorname{adj}(A)_{kj} = \sum_{k=1}^{n} A_{ik} (-1)^{k+j} \det(A^{jk}) = \begin{cases} \det(A), & \text{pro } i = j, \\ 0, & \text{pro } i \neq j. \end{cases}$$

Zdůvodnění poslední rovnosti je, že pro $i = j$ se jedná o Laplaceův rozvoj $\det(A)$ podle $j$-tého řádku. Pro $i \neq j$ se zase jedná o rozvoj podle $j$-tého řádku matice $A$, v níž ale nejprve $j$-tý řádek nahradíme $i$-tým. Tato matice bude mít dva stejné řádky a tím pádem nulový determinant.

</div>

Pro regulární matici $A$ je $\det(A) \neq 0$ a vydělením $\det(A)$ dostaneme explicitní vzoreček pro inverzní matici $A^{-1}$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(Důsledek 9.19)</span></p>

*Je-li $A \in \mathbb{T}^{n \times n}$ regulární, pak $A^{-1} = \frac{1}{\det(A)} \operatorname{adj}(A)$.*

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 9.20 — Adjungovaná matice)</span></p>

Buď

$$A = \begin{pmatrix} 1 & 2 & 3 \\ 1 & 2 & 1 \\ 2 & 5 & 5 \end{pmatrix}.$$

Pak:

$$\operatorname{adj}(A)_{12} = (-1)^{1+2} \begin{vmatrix} 2 & 3 \\ 5 & 5 \end{vmatrix} = 5, \quad \ldots$$

Celkem:

$$\operatorname{adj}(A) = \begin{pmatrix} 5 & 5 & -4 \\ -3 & -1 & 2 \\ 1 & -1 & 0 \end{pmatrix}.$$

Tedy:

$$A^{-1} = \frac{1}{\det(A)} \operatorname{adj}(A) = \frac{1}{2} \begin{pmatrix} 5 & 5 & -4 \\ -3 & -1 & 2 \\ 1 & -1 & 0 \end{pmatrix}.$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Poznámka 9.21 — Derivace determinantu)</span></p>

Uvažujme determinant jako funkci $\det(A) \colon \mathbb{R}^{n \times n} \to \mathbb{R}$. Problém nyní zní určit parciální derivaci $\det(A)$ podle $a_{ij}$ a sestavit matici parciálních derivací.

Pro tento účel vyjdeme z Laplaceova rozvoje $\det(A) = \sum_{k=1}^{n} (-1)^{i+k} a_{ik} \det(A^{ik})$ a jednoduše odvodíme

$$\frac{\partial \det(A)}{\partial a_{ij}} = (-1)^{i+j} \det(A^{ij}).$$

Tudíž matice parciálních derivací je $\partial \det(A) = \operatorname{adj}(A)^T$.

Použijeme-li konkrétně matici z příkladu 9.20, tak $\det(A) = 2$. Protože $\operatorname{adj}(A)_{33} = 0$, tak determinant matice $A$ se nezmění při změně prvku $a_{33}$. Na druhou stranu, při malém zvětšení prvku $a_{11}$ se determinant zvětší výrazně (protože $\operatorname{adj}(A)_{11} = 5$) a při zvětšení prvku $a_{13}$ se determinant zvětší méně výrazně (protože $\operatorname{adj}(A)_{31} = 1$).

</div>

### 9.4 Aplikace

Věta o adjungované matici dává následující charakterizaci celočíselnosti inverzní matice.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Tvrzení 9.22)</span></p>

*Buď $A \in \mathbb{Z}^{n \times n}$. Pak $A^{-1}$ má celočíselné hodnoty právě tehdy, když $\det(A) = \pm 1$.*

*Důkaz.* Implikace „$\Rightarrow$". Víme $1 = \det(A) \det(A^{-1})$. Jsou-li matice $A, A^{-1}$ celočíselné, pak i jejich determinanty jsou celočíselné a tudíž musejí být rovny $\pm 1$.

Implikace „$\Leftarrow$". Víme $A^{-1}_{ij} = \frac{1}{\det(A)} (-1)^{i+j} \det(A^{ji})$. To je celočíselná hodnota, jestliže $\det(A) = \pm 1$ a $\det(A^{ji})$ je celé číslo.

</div>

Další ukázka použití determinantu je v polynomech. Determinant z následující věty se nazývá *resultant* a používá se například k řešení nelineárních rovnic.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Tvrzení 9.23 — Resultant)</span></p>

*Polynomy $p(x) = a_n x^n + \ldots + a_1 x + a_0$, $q(x) = b_m x^m + \ldots + b_1 x + b_0$ mají společný kořen právě tehdy, když*

$$\begin{vmatrix} a_n & a_{n-1} & \ldots & & a_0 & & \\ & a_n & a_{n-1} & \ldots & a_0 & & \\ & & \ddots & \ddots & & \ddots & \\ & & & a_n & a_{n-1} & \ldots & a_0 \\ b_m & b_{m-1} & \ldots & b_1 & b_0 & & \\ & \ddots & \ddots & & & \ddots & \\ & & b_m & b_{m-1} & \ldots & & b_1 & b_0 \end{vmatrix} = 0.$$

</div>

#### Geometrická interpretace determinantu

Determinant má pěkný geometrický význam. Uvažujeme-li lineární zobrazení $x \mapsto Ax$ s maticí $A \in \mathbb{R}^{n \times n}$, pak geometrická tělesa mění v tomto zobrazení svůj objem s koeficientem $|\det(A)|$. Pojem „objem" v prostoru $\mathbb{R}^n$ nedefinujeme formálně a spoléháme na intuitivní představu. Objem běžných geometrických útvarů v prostoru $\mathbb{R}^1$ odpovídá délkám, v prostoru $\mathbb{R}^2$ odpovídá obsahu a v prostoru $\mathbb{R}^3$ odpovídá objemu v běžném významu.

Uvažujeme nejprve speciální případ rovnoběžnostěnu. *Rovnoběžnostěn* s lineárně nezávislými hranami $a_1, \ldots, a_m$ definujeme jako množinu $\lbrace x \in \mathbb{R}^n;\, x = \sum_{i=1}^{m} \alpha_i a_i, \, 0 \le \alpha_i \le 1 \rbrace$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Věta 9.24 — Objem rovnoběžnostěnu)</span></p>

*Buď $A \in \mathbb{R}^{m \times n}$ a uvažme rovnoběžnostěn s hranami danými řádky matice $A$. Pak jeho objem (jakožto $m$-dimenzionálního útvaru) je $\sqrt{\det(AA^T)}$. Speciálně, pro $m = n$ je objem $|\det(A)|$.*

*Důkaz.* Matematickou indukcí podle $m$. Pro $m = 1$ je to zřejmé, postupme k indukčnímu kroku. Označme $i$-tý řádek matice $A$ jako vektor $a_i^T$ a definujme matici $D$, která vznikne z $A$ odstraněním posledního řádku. Rozložme $a_m = b_m + c_m$, kde $c_m \in \mathcal{R}(D)$ a $b_m \in \mathcal{R}(D)^\perp$ podle poznámky 8.40. Řádky matice $D$ generují rovnoběžnostěn menší dimenze, jenž tvoří podstavu celkového rovnoběžnostěnu. Podle indukčního předpokladu je obsah podstavy $\sqrt{\det(DD^T)}$. Vektor $b_m$ je kolmý na podstavu a jeho délka odpovídá výšce $\|b_m\|$ rovnoběžnostěnu.

Dále,

$$A' A'^T = \begin{pmatrix} D \\ b_m^T \end{pmatrix} (D^T \quad b_m) = \begin{pmatrix} DD^T & Db_m \\ b_m^T D^T & b_m^T b_m \end{pmatrix} = \begin{pmatrix} DD^T & o \\ o^T & b_m^T b_m \end{pmatrix}$$

Tedy $\det(A' A'^T) = b_m^T b_m \det(DD^T)$ a odmocněním dostaneme

$$\sqrt{\det(A' A'^T)} = \|b_m\| \sqrt{\det(DD^T)}.$$

To odpovídá intuitivní představě objemu jako velikosti výšky krát obsah základny. Od $A'$ k $A$ lze přejít pomocí elementárních řádkových úprav, neboť k poslednímu řádku stačí přičíst $c_m$, což je lineární kombinace $a_1, \ldots, a_{m-1}$. Tedy existují elementární matice $E_1, \ldots, E_k$ tak, že $A = E_1 \ldots E_k A'$; navíc jejich determinant je 1, protože jen přičítají násobek řádku jednoho k jinému. Nyní

$$\det(AA^T) = \det(E_1 \ldots E_k A' A'^T E_k^T \ldots E_1^T) = \det(A' A'^T).$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Poznámka 9.25 — Objem rovnoběžnostěnu a elementární úpravy)</span></p>

Platnost věty 9.24 lze nahlédnout geometricky rozborem vlivu elementárních úprav. Uvažujme rovnoběžnostěn generovaný řádky matice $A \in \mathbb{R}^{n \times n}$ a chceme ukázat, že jeho objem je $|\det(A)|$. Víme, že determinant se nezmění pokud na matici provádíme třetí elementární úpravu (přičtení násobku jednoho řádku k jinému). Samotný rovnoběžnostěn se ale změní. Pochopit, proč objem zůstává zachován, je snadné z geometrického náhledu: přičtením násobku řádku k jinému (například poslednímu) znamená, že se rovnoběžnostěn zkosí či narovná, ale jak základna i výška zůstane stejná.

Představit si ostatní elementární úpravy je ještě snazší. Prohození řádků matice $A$ znamená překlopení rovnoběžnostěnu a jeho objem se proto nezmění. Vynásobení řádku matice $A$ číslem $\alpha$ pak protáhne rovnoběžnostěn v jednom směru, a tudíž se objem změní $\alpha$-krát. Vynásobení celé matice $A$ číslem $\alpha$ protáhne rovnoběžnostěn ve všech směrech a objem se změní $\alpha^n$-krát.

Je-li matice $A$ singulární, pak odpovídající rovnoběžnostěn leží v nějakém podprostoru dimenze menší než $n$, a tudíž je jeho objem nulový. Je-li matice $A$ regulární, pak ji elementárními úpravami převedeme na jednotkovou matici — odpovídající rovnoběžnostěn je jednotková krychle, a ta má objem 1.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Poznámka 9.26 — Vysvětlení definice determinantu)</span></p>

Předchozí poznámka umožňuje alternativní způsob zavedení determinantu a vysvětlení jeho definice. Kdybychom chtěli zavést determinant matice $A$ jako objem odpovídajícího rovnoběžnostěnu, narazíme na problém znaménka, protože objem je vždy nezáporný. Zavedeme tedy něco jako orientovaný objem, a to pomocí základních vlastností, které by objem měl splňovat:

1. Determinant jednotkové matice $I_n$ je roven 1, což odpovídá objemu jednotkové krychle.
2. Výměna řádků změní znaménko determinantu. To odpovídá vlastnosti, že objem rovnoběžnostěnu se nezmění změnou pořadí hran, tedy překlopením, nicméně změna znaménka zavede právě určitou orientaci do definice determinantu.
3. Vynásobení řádku matice $A$ číslem $\alpha \in \mathbb{R}$ změní determinant s koeficientem $\alpha$. To odpovídá protažení rovnoběžnostěnu ve směru dané hrany, a tím pádem odpovídající změnu objemu.
4. Řádková linearita determinantu ve smyslu věty 9.5. Důsledek této vlastnosti je například to, že zkosení nezmění objem rovnoběžnostěnu, a proto i determinant zůstane stejný.

Z těchto základních vlastností již jdou odvodit všechny ostatní vlastnosti determinantu a vysvětlit i původní definici $\det(A) = \sum_{p \in S_n} \operatorname{sgn}(p) a_{1,p(1)} \ldots a_{n,p(n)}$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Poznámka 9.27 — Objem jiných geometrických těles)</span></p>

Buď $A \in \mathbb{R}^{n \times n}$. Jak jsme již zmínili, objem geometrických těles se při zobrazení $x \mapsto Ax$ mění s koeficientem $|\det(A)|$. Krychle o hraně 1 se zobrazí na rovnoběžnostěn o hranách, které odpovídají sloupcům matice $A$, a jeho objem je proto $|\det(A^T)| = |\det(A)|$.

Tuto vlastnost můžeme zobecnit na ostatní běžně používaná geometrická tělesa, jako je koule, elipsoid, mnohostěn atp. Takové těleso lze totiž pokrýt krychlemi, a jeho obraz je tedy aproximován rovnoběžnostěny a změna objemu je přibližně $|\det(A)|$. Postupným zjemňováním aproximace (zmenšováním krychlí) dostaneme limitním přechodem výsledný poměr.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Poznámka 9.29 — Substituce u vícerozměrných integrálů)</span></p>

Geometrická interpretace determinantu rovněž umožňuje snadno nahlédnout platnost věty o substituci ve vícerozměrných integrálech. Věta za celkem obecných předpokladů říká, že

$$\int_{\varphi(M)} f(y) \, \mathrm{d}y = \int_M f(\varphi(x)) \cdot |\det(\nabla \varphi(x))| \, \mathrm{d}x,$$

kde $M \subseteq \mathbb{R}^n$ je otevřená množina, $\varphi \colon M \to \mathbb{R}^n$ je prostá funkce se spojitými parciálními derivacemi a $\nabla \varphi(x)$ je Jacobiho matice parciálních derivací funkce $\varphi(x)$ (viz poznámka 6.30), která musí být regulární pro všechna $x \in M$. Vysvětlení rovnosti je pak zřejmé z geometrického náhledu. Zobrazení $\varphi(x)$ sice není lineární, ale lokálně lze linearizovat právě Jacobiho maticí $\nabla \varphi(x)$. Zobrazení pak lokálně mění objemy s koeficientem, který odpovídá determinantu Jacobiho matice. Tudíž i integrál se mění se stejným faktorem.

</div>

Determinanty se používají při řešení ještě mnoha dalších geometrických problémů. Výpočtem determinantu tak například snadno rozhodneme, zda daný bod v rovině leží uvnitř či vně kružnice zadané svými třemi body, a podobně ve vyšších dimenzích.

Zmiňme úlohu, která souvisí s objemem rovnoběžnostěnu, a to určení objemu mnohostěnu s $n + 1$ vrcholy v $\mathbb{R}^n$. Bez újmy na obecnost nechť je jeden vrchol $a_0$ v počátku a ostatní mají pozice $a_1, \ldots, a_n \in \mathbb{R}^n$. Definujme matici $A \in \mathbb{R}^{n \times n}$ tak, že její sloupce jsou vektory $a_1, \ldots, a_n$. Pak objem mnohostěnu je $\frac{1}{n!} |\det(A)|$, čili tvoří jen část rovnoběžnostěnu danou faktorem $1 : n!$.

Předchozí způsob výpočtu objemu mnohostěnu předpokládal, že známe pozice jednotlivých vrcholů v prostoru $\mathbb{R}^n$. V některých případech (např. molekulární biologie) jsou ale známy pouze vzdálenosti $d_{ij} = \|a_i - a_j\|$ mezi jednotlivými vrcholy, tj. délky hran mnohostěnu. V tomto případě spočítáme objem pomocí tzv. Cayleyho–Mengerova determinantu jako

$$\frac{(-1)^{n-1}}{2^n (n!)^2} \begin{vmatrix} 0 & 1 & 1 & 1 & \ldots & 1 \\ 1 & 0 & d_{01}^2 & d_{02}^2 & \ldots & d_{0n}^2 \\ 1 & d_{10}^2 & 0 & d_{12}^2 & \ldots & d_{1n}^2 \\ 1 & d_{20}^2 & d_{21}^2 & 0 & \ldots & d_{2n}^2 \\ \vdots & \vdots & \vdots & \vdots & \ddots & \vdots \\ 1 & d_{n0}^2 & d_{n1}^2 & d_{n2}^2 & \ldots & 0 \end{vmatrix}.$$

### Shrnutí ke kapitole 9

Determinant matice $A$ je číselná charakteristika matice a lze spočítat efektivně pomocí Gaussovy eliminace — stačí jen určit, jak elementární úpravy mění determinant matice. Determinant mj. udává, zda matice je regulární či singulární, a pomocí determinantu můžeme také explicitně vyjádřit řešení soustavy $Ax = b$ s regulární maticí $A$. Podobně můžeme explicitně vyjádřit inverzi regulární matice, což vede na pojem adjungovaná matice. Geometricky pak determinant reprezentuje koeficient, se kterým se mění objem těles při lineárním zobrazení $x \mapsto Ax$; speciálně udává objem rovnoběžnostěnu jehož hrany jsou dané řádky matice $A$.

## Kapitola 10 — Vlastní čísla

Vlastní čísla (dříve též nazývaná „charakteristická čísla"), podobně jako determinant, představují určitou charakteristiku matice. Poskytují o matici a o odpovídajícím lineárním zobrazení mnoho důležitých informací.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Definice 10.1 — Vlastní čísla a vlastní vektory)</span></p>

Buď $A \in \mathbb{C}^{n \times n}$. Pak $\lambda \in \mathbb{C}$ je *vlastní číslo* matice $A$ a $x \in \mathbb{C}^n$ jemu příslušný *vlastní vektor*, pokud $Ax = \lambda x$, $x \neq o$.

</div>

Podmínka $x \neq o$ je nezbytná, protože pro $x = o$ by rovnost byla triviálně splněna pro každé $\lambda \in \mathbb{C}$. Na druhou stranu, $\lambda = 0$ klidně může nastat. Vlastní vektor při daném vlastním čísle není určen jednoznačně — každý jeho nenulový násobek je také vlastním vektorem. Někdy se proto vlastní vektor normuje tak, aby $\|x\| = 1$.

Přirozeně, vlastní čísla a vektory lze definovat stejně nad jakýmkoli jiným tělesem. My zůstaneme u $\mathbb{R}$ resp. $\mathbb{C}$. Jak uvidíme později, komplexním číslům se nevyhneme i když matice $A$ je reálná.

Vlastní čísla se dají zavést i obecněji. Buď $V$ vektorový prostor a $f \colon V \to V$ lineární zobrazení. Pak $\lambda$ je vlastní číslo a $x \neq o$ příslušný vlastní vektor, pokud platí $f(x) = \lambda x$. My se však vesměs budeme zabývat vlastními čísly matic, protože vzhledem k maticové reprezentaci lineárních zobrazení můžeme úlohu hledání vlastních čísel a vektorů lineárních zobrazení na konečně generovaných prostorech redukovat na matice.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 10.2 — Geometrická interpretace vlastních čísel a vektorů)</span></p>

Vlastní vektor reprezentuje invariantní směr při zobrazení $x \mapsto Ax$, tedy směr, který se zobrazí opět na ten samý směr. Jinými slovy, je-li $v$ vlastní vektor, pak přímka $\operatorname{span}\lbrace v \rbrace$ se zobrazí do sebe sama. Vlastní číslo pak představuje škálování v tomto invariantním směru.

- Překlopení dle přímky $y = -x$, matice zobrazení $A = \begin{pmatrix} 0 & -1 \\ -1 & 0 \end{pmatrix}$: vlastní číslo 1, vlastní vektor $(-1, 1)^T$; vlastní číslo $-1$, vlastní vektor $(1, 1)^T$.
- Rotace o úhel $90°$, matice zobrazení $A = \begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix}$: žádná reálná vlastní čísla.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Věta 10.3 — Charakterizace vlastních čísel a vektorů)</span></p>

Buď $A \in \mathbb{C}^{n \times n}$. Pak

1. $\lambda \in \mathbb{C}$ je vlastním číslem $A$ právě tehdy, když $\det(A - \lambda I_n) = 0$,
2. $x \in \mathbb{C}^n$ je vlastním vektorem příslušným k vlastnímu číslu $\lambda \in \mathbb{C}$ právě tehdy, když $o \neq x \in \operatorname{Ker}(A - \lambda I_n)$.

</div>

*Důkaz.* (1) $\lambda \in \mathbb{C}$ je vlastním číslem $A$ právě tehdy, když $Ax = \lambda I_n x$, $x \neq o$, neboli $(A - \lambda I_n)x = o$, $x \neq o$, což je ekvivalentní singularitě matice $A - \lambda I_n$, a to zase podmínce $\det(A - \lambda I_n) = 0$. (2) Analogicky, $x \in \mathbb{C}^n$ je vlastním vektorem k vlastnímu číslu $\lambda \in \mathbb{C}$ právě tehdy, když $(A - \lambda I_n)x = o$, $x \neq o$, tedy $x$ je v jádru matice $A - \lambda I_n$.

Důsledkem věty je, že k danému vlastnímu číslu $\lambda$ přísluší $\dim \operatorname{Ker}(A - \lambda I_n) = n - \operatorname{rank}(A - \lambda I_n)$ lineárně nezávislých vlastních vektorů.

### Charakteristický polynom

První část věty 10.3 říká, že $\lambda \in \mathbb{C}$ je vlastním číslem matice $A$ právě tehdy, když matice $A - \lambda I_n$ je singulární, neboli $\det(A - \lambda I_n) = 0$. Pokud se na $\lambda$ díváme jako na komplexní proměnnou, tak najít vlastní číslo je totéž jako najít řešení rovnice $\det(A - \lambda I_n) = 0$. Rozepsáním determinantu z definice dostaneme polynom stupně nanejvýš $n$ v proměnné $\lambda$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Definice 10.4 — Charakteristický polynom)</span></p>

*Charakteristický polynom* matice $A \in \mathbb{C}^{n \times n}$ vzhledem k proměnné $\lambda$ je $p_A(\lambda) = \det(A - \lambda I_n)$.

</div>

Z definice determinantu je patrné, že charakteristický polynom se dá vyjádřit ve tvaru

$$p_A(\lambda) = \det(A - \lambda I_n) = (-1)^n \lambda^n + a_{n-1}\lambda^{n-1} + \ldots + a_1 \lambda + a_0.$$

Tedy je to skutečně polynom a má stupeň $n$. Snadno nahlédneme, že $a_{n-1} = (-1)^{n-1}(a_{11} + \ldots + a_{nn})$ a po dosazení $\lambda = 0$ získáme $a_0 = \det(A)$.

Podle základní věty algebry má tento polynom $n$ komplexních kořenů (včetně násobností), označme je $\lambda_1, \ldots, \lambda_n$. Pak

$$p_A(\lambda) = (-1)^n (\lambda - \lambda_1) \ldots (\lambda - \lambda_n).$$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Věta 10.5)</span></p>

Vlastní čísla matice $A \in \mathbb{C}^{n \times n}$ jsou právě kořeny jejího charakteristického polynomu $p_A(\lambda)$, a je jich $n$ včetně násobností.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 10.6)</span></p>

Mějme matici $A = \begin{pmatrix} 0 & -2 \\ 2 & 0 \end{pmatrix}$ obdobnou matici z příkladu 10.2. Pak

$$p_A(\lambda) = \det(A - \lambda I_n) = \det \begin{pmatrix} -\lambda & -2 \\ 2 & -\lambda \end{pmatrix} = \lambda^2 + 4.$$

Kořeny polynomu, a tedy vlastními čísly matice $A$, jsou $\pm 2i$. Vlastní vektor příslušný $2i$ je $(1, -i)^T$ a vlastní vektor příslušný $-2i$ je $(1, i)^T$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Definice 10.7 — Spektrum a spektrální poloměr)</span></p>

Nechť $A \in \mathbb{C}^{n \times n}$ má vlastní čísla $\lambda_1, \ldots, \lambda_n$. Pak *spektrum* matice $A$ je množina jejích vlastních čísel $\lbrace \lambda_1, \ldots, \lambda_n \rbrace$ a *spektrální poloměr* je $\rho(A) = \max_{i=1,\ldots,n} |\lambda_i|$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Definice 10.8 — Algebraická a geometrická násobnost vlastního čísla)</span></p>

Buď $\lambda \in \mathbb{C}$ vlastní číslo matice $A \in \mathbb{C}^{n \times n}$. *Algebraická násobnost* $\lambda$ je rovna násobnosti $\lambda$ jakožto kořene $p_A(\lambda)$. *Geometrická násobnost* $\lambda$ je rovna $n - \operatorname{rank}(A - \lambda I_n)$, tj. počtu lineárně nezávislých vlastních vektorů, které odpovídají $\lambda$.

</div>

Algebraická násobnost je vždy větší nebo rovna geometrické násobnosti, což vyplyne v sekci 10.4.

Počítat vlastní čísla jako kořeny charakteristického polynomu není příliš efektivní. Již jenom určit jednotlivé koeficienty tohoto polynomu není triviální úkol. Navíc, jak víme, pro kořeny polynomu neexistuje žádný vzoreček ani konečný postup a počítají se iterativními metodami. Totéž platí i o vlastních číslech. Nicméně pro některé speciální matice, jako jsou třeba trojúhelníkové matice, můžeme vlastní čísla určit snadno.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 10.9 — Vlastní čísla trojúhelníkové matice)</span></p>

- Nechť $A \in \mathbb{C}^{n \times n}$ je trojúhelníková matice. Pak její vlastní čísla jsou prvky na diagonále, neboť $\det(A - \lambda I_n) = (a_{11} - \lambda) \ldots (a_{nn} - \lambda)$.
- Speciálně, $I_n$ má vlastní číslo 1, které je $n$-násobné. Množina příslušných vlastních vektorů je $\mathbb{R}^n \setminus \lbrace o \rbrace$.
- Speciálně, $0_n$ má vlastní číslo 0, které je $n$-násobné. Množina příslušných vlastních vektorů je $\mathbb{R}^n \setminus \lbrace o \rbrace$.
- Speciálně, matice $\begin{pmatrix} 1 & 1 \\ 0 & 1 \end{pmatrix}$ má vlastní číslo 1, které je dvojnásobné (algebraicky). Odpovídající vlastní vektor je až na násobek pouze $(1, 0)^T$, proto je geometrická násobnost vlastního čísla 1 pouze jedna.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 10.10 — Geometrická interpretace: skosení a protáhnutí)</span></p>

Mějme matici $A = \begin{pmatrix} 1.5 & 0.75 \\ 0 & 1 \end{pmatrix}$. Příslušné lineární zobrazení $x \mapsto Ax$ geometricky představuje skosení a protáhnutí v ose $x_1$ o $50\%$, ve směru osy $x_2$ nijak neprotahuje.

Vlastní čísla matice $A$ jsou 1.5 a 1, a jim příslušející vlastní vektory jsou $(1, 0)^T$ a $(-1.5, 1)^T$. První vlastní číslo a vektor říkají, že se obrázek protáhne o $50\%$ ve směru osy $x_1$. Druhé vlastní číslo a vektor říkají, že se obrázek ve směru vektoru $(-1.5, 1)^T$ nedeformuje.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Tvrzení 10.11 — Součin a součet vlastních čísel)</span></p>

Buď $A \in \mathbb{C}^{n \times n}$ s vlastními čísly $\lambda_1, \ldots, \lambda_n$. Pak

1. $\det(A) = \lambda_1 \ldots \lambda_n$,
2. $\operatorname{trace}(A) = \lambda_1 + \ldots + \lambda_n$.

</div>

*Důkaz.* (1) Víme, že $p_A(\lambda) = (-1)^n(\lambda - \lambda_1) \ldots (\lambda - \lambda_n)$. Dosazením $\lambda = 0$ dostáváme $\det(A) = (-1)^n(-\lambda_1) \ldots (-\lambda_n) = \lambda_1 \ldots \lambda_n$. (2) Porovnáním koeficientů u $\lambda^{n-1}$ v různých vyjádřeních charakteristického polynomu.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Tvrzení 10.12 — Vlastnosti vlastních čísel)</span></p>

Nechť $A \in \mathbb{C}^{n \times n}$ má vlastní čísla $\lambda_1, \ldots, \lambda_n$ a jim odpovídající vlastní vektory $x_1, \ldots, x_n$. Pak:

1. $A$ je regulární právě tehdy, když $0$ není její vlastní číslo,
2. je-li $A$ regulární, pak $A^{-1}$ má vlastní čísla $\lambda_1^{-1}, \ldots, \lambda_n^{-1}$ a vlastní vektory $x_1, \ldots, x_n$,
3. $A^2$ má vlastní čísla $\lambda_1^2, \ldots, \lambda_n^2$ a vlastní vektory $x_1, \ldots, x_n$,
4. $\alpha A$ má vlastní čísla $\alpha \lambda_1, \ldots, \alpha \lambda_n$ a vlastní vektory $x_1, \ldots, x_n$,
5. $A + \alpha I_n$ má vlastní čísla $\lambda_1 + \alpha, \ldots, \lambda_n + \alpha$ a vlastní vektory $x_1, \ldots, x_n$,
6. $A^T$ má vlastní čísla $\lambda_1, \ldots, \lambda_n$, ale vlastní vektory obecně jiné.

</div>

*Důkaz.* (1) $A$ má vlastní číslo 0 právě tehdy, když $0 = \det(A - 0 \cdot I_n) = \det(A)$, neboli když $A$ je singulární. (2) Pro každé $i$ je $Ax_i = \lambda_i x_i$. Přenásobením $A^{-1}$ dostaneme $x_i = \lambda_i A^{-1} x_i$ a vydělením $\lambda_i \neq 0$ pak $A^{-1} x_i = \lambda_i^{-1} x_i$.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 10.13)</span></p>

Uvažujme matice

$$A = \begin{pmatrix} 0 & 1 \\ 0 & 0 \end{pmatrix}, \quad B = \begin{pmatrix} 0 & 0 \\ 1 & 0 \end{pmatrix}.$$

Obě mají všechna vlastní čísla nulová. Součet matic

$$A + B = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$$

má vlastní čísla $-1$ a $1$. Součin matic

$$AB = \begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix}$$

má vlastní čísla 0 a 1. Z tohoto jednoduchého příkladu se dá usuzovat, že sčítáním a násobením matic se vlastní čísla mění a nedají se snadno odhadovat z vlastních čísel původních matic. Matice určitého typu (např. diagonalizovatelné, sekce 10.3) se chovají rozumněji a součinem či součtem takových matic nemohou vlastní čísla narůst zcela libovolně.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Poznámka 10.14 — Topologie množiny regulárních matic)</span></p>

Množina regulárních matic je tzv. *hustá množina* v prostoru $\mathbb{R}^{n \times n}$. To znamená, že každá matice $A \in \mathbb{R}^{n \times n}$ se dá vyjádřit jako limita vhodné posloupnosti regulárních matic. Pro regulární matici $A$ je pozorování zřejmé, stačí uvažovat posloupnost složenou z matice $A$. Je-li $A$ singulární, pak $A + \frac{1}{k}I_n$ je regulární pro dost velké $k$, neboť nebude mít nulové vlastní číslo. Tudíž máme posloupnost regulárních matic $A + \frac{1}{k}I_n$ konvergující k matici $A$ pro $k \to \infty$.

Množina regulárních matic je i tzv. *otevřená množina* v prostoru $\mathbb{R}^{n \times n}$ (a tím pádem množina singulárních matic je uzavřená). Tato vlastnost říká, že pro každou regulární matici $A \in \mathbb{R}^{n \times n}$ jsou i matice v jejím okolí regulární. Tvrzení nahlédneme z toho, že $\det(A) \neq 0$ a že determinant je spojitá funkce. Tudíž $\det(A') \neq 0$ i pro matice $A'$ z dostatečně malého okolí kolem matice $A$.

</div>

Víme, že i reálná matice může mít některá vlastní čísla komplexní. Ta komplexní vlastní čísla ale vždy můžeme spárovat do dvojic navzájem komplexně sdružených čísel.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Tvrzení 10.15)</span></p>

Je-li $\lambda \in \mathbb{C}$ vlastní číslo matice $A \in \mathbb{R}^{n \times n}$, pak i komplexně sdružené $\overline{\lambda}$ je vlastním číslem $A$.

</div>

*Důkaz.* Víme, že $\lambda$ je kořenem $p_A(\lambda) = (-1)^n \lambda^n + a_{n-1}\lambda^{n-1} + \ldots + a_1\lambda + a_0 = 0$. Komplexním sdružením obou stran rovnosti máme $(-1)^n \overline{\lambda}^n + a_{n-1}\overline{\lambda}^{n-1} + \ldots + a_1\overline{\lambda} + a_0 = 0$, tedy i $\overline{\lambda}$ je kořenem $p_A(\lambda)$.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 10.16)</span></p>

Spektrum reálné matice je tedy množina symetrická podle reálné osy. Komplexní matice mohou mít za spektrum jakýchkoli $n$ komplexních čísel.

</div>

Nyní ukážeme, že výpočet kořenů polynomu lze převést na úlohu hledání vlastních čísel určité matice.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Definice 10.17 — Matice společnice)</span></p>

Buď $p(x) = x^n + a_{n-1}x^{n-1} + \ldots + a_1 x + a_0$. Pak *matice společnice* polynomu $p(x)$ je čtvercová matice řádu $n$ definovaná

$$C(p) \coloneqq \begin{pmatrix} 0 & \ldots & \ldots & 0 & -a_0 \\ 1 & \ddots & & \vdots & -a_1 \\ 0 & \ddots & \ddots & \vdots & -a_2 \\ \vdots & \ddots & \ddots & 0 & \vdots \\ 0 & \ldots & 0 & 1 & -a_{n-1} \end{pmatrix}.$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Věta 10.18 — O matici společnici)</span></p>

Pro charakteristický polynom matice $C(p)$ platí $p_{C(p)}(\lambda) = (-1)^n p(\lambda)$, tedy vlastní čísla matice $C(p)$ odpovídají kořenům polynomu $p(\lambda)$.

</div>

Věta má mj. za důsledek, že úloha hledání kořenů reálných polynomů a vlastních čísel matic jsou na sebe navzájem převoditelné: věta 10.5 redukuje hledání vlastních čísel matice na hledání kořenů polynomu, a věta 10.18 to dělá naopak.

### Cayleyho–Hamiltonova věta

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 10.19 — Polynomiální matice)</span></p>

Abychom lépe porozuměli následující větě, uvádíme příklad polynomiální matice a maticového polynomu

$$\begin{pmatrix} \lambda^2 - \lambda & 2\lambda - 3 \\ 7 & 5\lambda^2 - 4 \end{pmatrix} = \lambda^2 \begin{pmatrix} 1 & 0 \\ 0 & 5 \end{pmatrix} + \lambda \begin{pmatrix} -1 & 2 \\ 0 & 0 \end{pmatrix} + \begin{pmatrix} 0 & -3 \\ 7 & -4 \end{pmatrix}.$$

Jsou to dva zápisy stejné matice s parametrem $\lambda$, a můžeme jednoduše převést jeden zápis na druhý.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Věta 10.20 — Cayleyho–Hamiltonova)</span></p>

Buď $A \in \mathbb{C}^{n \times n}$ a $p_A(\lambda) = (-1)^n \lambda^n + a_{n-1}\lambda^{n-1} + \ldots + a_1\lambda + a_0$. Pak

$$(-1)^n A^n + a_{n-1}A^{n-1} + \ldots + a_1 A + a_0 I_n = 0.$$

</div>

*Důkaz.* Víme, že pro adjungované matice platí $(A - \lambda I_n) \operatorname{adj}(A - \lambda I_n) = \det(A - \lambda I_n)I_n$. Každý prvek $\operatorname{adj}(A - \lambda I_n)$ je polynom stupně nanejvýš $n - 1$ proměnné $\lambda$, takže se dá vyjádřit ve tvaru $\operatorname{adj}(A - \lambda I_n) = \lambda^{n-1}B_{n-1} + \ldots + \lambda B_1 + B_0$ pro určité $B_{n-1}, \ldots, B_0 \in \mathbb{C}^{n \times n}$. Dosazením a porovnáním koeficientů u mocnin $\lambda$ dostaneme soustavu rovnic, jejichž sečtením po vynásobení odpovídajícími mocninami $A$ dostaneme požadovaný výsledek.

Zkráceně můžeme tvrzení Cayleyho–Hamiltonovy věty vyjádřit jako $p_A(A) = 0$, tj. matice je sama kořenem svého charakteristického polynomu.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(Důsledek 10.21)</span></p>

Buď $A \in \mathbb{C}^{n \times n}$. Pak:

1. Pro každé $k \in \mathbb{N}$ je $A^k \in \operatorname{span}\lbrace I_n, A, \ldots, A^{n-1} \rbrace$, tedy $A^k$ je lineární kombinací matic $I_n, A, \ldots, A^{n-1}$.
2. Je-li $A$ regulární, pak $A^{-1} \in \operatorname{span}\lbrace I_n, A, \ldots, A^{n-1} \rbrace$.

</div>

*Důkaz.* (1) Stačí uvažovat $k \ge n$. Při dělení polynomu $\lambda^k$ polynomem $p_A(\lambda)$ se zbytkem tak vlastně polynom $\lambda^k$ rozložíme $\lambda^k = r(\lambda) p_A(\lambda) + s(\lambda)$. Pak $A^k = r(A) p_A(A) + s(A) = s(A) = b_{n-1}A^{n-1} + \ldots + b_1 A + b_0 I_n$. (2) Z $p_A(A) = (-1)^n A^n + a_{n-1}A^{n-1} + \ldots + a_1 A + a_0 I_n = 0$ a $a_0 \neq 0$ z regularity $A$.

Podle tohoto důsledku lze velkou mocninu $A^k$ matice $A$ spočítat alternativně tak, že najdeme příslušné koeficienty charakteristického polynomu a vyjádříme $A^k$ jako lineární kombinaci $I_n, A, \ldots, A^{n-1}$. Podobně můžeme vyjádřit i $A^{-1}$, a tím pádem řešení soustavy $Ax = b$ s regulární maticí jako $A^{-1}b = \frac{1}{a_0}(-(-1)^n A^{n-1}b - \ldots - a_1 b)$.

### Diagonalizovatelnost

Při řešení soustav lineárních rovnic pomocí Gaussovy–Jordanovy eliminace jsme používali elementární řádkové úpravy. Ty nemění množinu řešení a upraví matici soustavy na tvar, ze kterého snadno vyčteme řešení. Je přirozené hledat analogické, spektrum neměnící, úpravy i na problém počítání vlastních čísel. Elementární řádkové úpravy použít nelze, protože ty mění spektrum. Vhodnou transformací je tzv. *podobnost*, protože ta spektrum nemění. A pokud se nám podaří touto transformací převést matici na diagonální tvar, máme vyhráno — na diagonále jsou hledaná vlastní čísla.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Definice 10.22 — Podobnost)</span></p>

Matice $A, B \in \mathbb{C}^{n \times n}$ jsou *podobné*, pokud existuje regulární $S \in \mathbb{C}^{n \times n}$ tak, že $A = SBS^{-1}$.

</div>

Podobnost jde ekvivalentně definovat vztahem $AS = SB$ pro nějakou regulární matici $S$.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 10.23)</span></p>

Matice $\begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix}$ a $\begin{pmatrix} 0 & 0 \\ 0 & 1 \end{pmatrix}$ jsou si podobné skrze matici $S = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Věta 10.24 — Vlastní čísla podobných matic)</span></p>

Podobné matice mají stejná vlastní čísla.

</div>

*Důkaz.* Z podobnosti matic existuje regulární $S$ taková, že $A = SBS^{-1}$. Pak

$$p_A(\lambda) = \det(A - \lambda I_n) = \det(SBS^{-1} - \lambda S I_n S^{-1}) = \det(S(B - \lambda I_n)S^{-1}) = \det(S)\det(B - \lambda I_n)\det(S^{-1}) = \det(B - \lambda I_n) = p_B(\lambda).$$

Obě matice mají stejné charakteristické polynomy, tedy i vlastní čísla.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 10.25)</span></p>

Ukažte, že podobnost jako binární relace je reflexivní, symetrická a tranzitivní. Tedy jedná se o relaci ekvivalenci.

</div>

Věta neříká nic o vlastních vektorech, ty se měnit mohou. Co ale zůstává neměnné, je jejich počet lineárně nezávislých vlastních vektorů.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Tvrzení 10.26)</span></p>

Nechť matice $A, B \in \mathbb{C}^{n \times n}$ jsou podobné a nechť je jejich vlastní číslo $\lambda$. Pak počet vlastních vektorů, odpovídajících $\lambda$, je stejný u obou matic.

</div>

*Důkaz.* Buď $A = SBS^{-1}$. Protože hodnost matice se nemění přenásobením regulární maticí, tak $\operatorname{rank}(A - \lambda I_n) = \operatorname{rank}(S(B - \lambda I_n)S^{-1}) = \operatorname{rank}(B - \lambda I_n)$. Tudíž dimenze jádra obou matic $A - \lambda I_n$ a $B - \lambda I_n$ jsou stejné, a tím pádem i počet vlastních vektorů.

Vlastní čísla se podobnostní transformací nemění, tedy jestliže matici $A$ převedeme podobnostní transformací na diagonální či obecněji trojúhelníkovou, tak na diagonále najdeme její vlastní čísla. Speciálně ty matice, které jdou převést na diagonální tvar, mají obzvlášť pěkné vlastnosti.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Definice 10.27 — Diagonalizovatelnost)</span></p>

Matice $A \in \mathbb{C}^{n \times n}$ je *diagonalizovatelná*, pokud je podobná nějaké diagonální matici.

</div>

Diagonalizovatelná matice $A$ jde tedy vyjádřit ve tvaru $A = S \Lambda S^{-1}$, kde $S$ je regulární a $\Lambda$ diagonální. Tomuto tvaru se říká *spektrální rozklad*, a to proto, že na diagonále matice $\Lambda$ je spektrum matice $A$.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 10.28)</span></p>

Ne každá matice je diagonalizovatelná, např. $A = \begin{pmatrix} 0 & 1 \\ 0 & 0 \end{pmatrix}$. Její vlastní číslo (dvojnásobné) je 0. Pokud by $A$ byla diagonalizovatelná, pak by byla podobná nulové matici, tedy $A = S 0 S^{-1} = 0$, což je spor.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Věta 10.29 — Charakterizace diagonalizovatelnosti)</span></p>

Matice $A \in \mathbb{C}^{n \times n}$ je diagonalizovatelná právě tehdy, když má $n$ lineárně nezávislých vlastních vektorů.

</div>

*Důkaz.* Implikace „$\Rightarrow$": Je-li $A$ diagonalizovatelná, pak má spektrální rozklad $A = S \Lambda S^{-1}$, kde $S$ je regulární a $\Lambda$ diagonální. Rovnost $AS = S\Lambda$ a porovnáním $j$-tých sloupců dostaneme $AS_{*j} = (S\Lambda)_{*j} = S\Lambda_{jj}e_j = \Lambda_{jj}S_{*j}$. Tedy $\Lambda_{jj}$ je vlastní číslo a $S_{*j}$ je příslušný vlastní vektor. Sloupce matice $S$ jsou lineárně nezávislé díky její regularitě.

Implikace „$\Leftarrow$": Nechť $A$ má vlastní čísla $\lambda_1, \ldots, \lambda_n$ a jim přísluší lineárně nezávislé vlastní vektory $x_1, \ldots, x_n$. Sestavme regulární matici $S \coloneqq (x_1 \mid \cdots \mid x_n)$ a diagonální $\Lambda \coloneqq \operatorname{diag}(\lambda_1, \ldots, \lambda_n)$. Pak $(AS)_{*j} = Ax_j = \lambda_j x_j = \Lambda_{jj} S_{*j} = (S\Lambda)_{*j}$. Tedy $AS = S\Lambda$, z čehož $A = S\Lambda S^{-1}$.

Nediagonalizovatelné matice jsou ty, pro které nastávají určité patologické situace, ale diagonalizovatelné matice mají celou řadu přirozených vlastností. Je-li matice $A$ diagonalizovatelná, pak:

- Algebraická a geometrická násobnost každého vlastního čísla $A$ je stejná.
- Hodnost matice $A$ je rovna počtu nenulových vlastních čísel $A$.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Poznámka 10.30 — Geometrická interpretace diagonalizace)</span></p>

Jiný pohled na diagonalizaci je geometrický: víme, že vlastní vektor představuje invariantní směr při zobrazení $x \mapsto Ax$. Nyní si představme, že $A$ představuje matici nějakého lineárního zobrazení $f \colon \mathbb{C}^n \to \mathbb{C}^n$ vzhledem k bázi $B$. Buď $S = {}_{B'}[id]_B$ matice přechodu od $B$ k jiné bázi $B'$. Pak $SAS^{-1} = {}_{B'}[id]_B \cdot {}_B[f]_B \cdot {}_B[id]_{B'} = {}_{B'}[f]_{B'}$ je matice zobrazení $f$ vzhledem k nové bázi $B'$. Nyní diagonalizovatelnost můžeme chápat jako hledání vhodné báze $B'$, aby příslušná matice byla diagonální, a tak jednoduše popisovala chování zobrazení.

Díky tomuto geometrickému pohledu snadno nahlédneme platnost věty 10.24. Podobnost znamená změnu báze, ale nemění samotné lineární zobrazení $f$, takže vlastní čísla musí zůstat stejná.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 10.31 — Geometrická interpretace diagonalizace)</span></p>

Buď $A = \begin{pmatrix} 3 & 1 \\ 1 & 3 \end{pmatrix}$. Vlastní čísla a vlastní vektory matice $A$ jsou: $\lambda_1 = 4$, $x_1 = (1, 1)^T$; $\lambda_2 = 2$, $x_2 = (-1, 1)^T$.

Diagonalizace má tvar:

$$A = S \Lambda S^{-1} = \begin{pmatrix} 1 & -1 \\ 1 & 1 \end{pmatrix} \begin{pmatrix} 4 & 0 \\ 0 & 2 \end{pmatrix} \begin{pmatrix} \frac{1}{2} & \frac{1}{2} \\ -\frac{1}{2} & \frac{1}{2} \end{pmatrix}.$$

Geometrická interpretace: V souřadném systému vlastních vektorů je matice zobrazení diagonální a zobrazení představuje jen škálování na osách.

</div>

Nyní ukážeme, že různým vlastním číslům odpovídají lineárně nezávislé vlastní vektory.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Tvrzení 10.32 — Vlastní vektory různých vlastních čísel)</span></p>

Nechť $\lambda_1, \ldots, \lambda_k$ jsou navzájem různá vlastní čísla (ne nutně všechna) matice $A \in \mathbb{C}^{n \times n}$. Pak odpovídající vlastní vektory $x_1, \ldots, x_k$ jsou lineárně nezávislé.

</div>

*Důkaz.* Matematickou indukcí podle $k$. Pro $k = 1$ zřejmé, neboť vlastní vektor je nenulový. Indukční krok $k \leftarrow k - 1$. Uvažme lineární kombinaci $\alpha_1 x_1 + \ldots + \alpha_k x_k = o$. Pak přenásobením maticí $A$ dostaneme $\alpha_1 \lambda_1 x_1 + \ldots + \alpha_k \lambda_k x_k = o$. Odečtením $\lambda_k$-násobku první rovnice od druhé dostaneme $\alpha_1(\lambda_1 - \lambda_k) x_1 + \ldots + \alpha_{k-1}(\lambda_{k-1} - \lambda_k) x_{k-1} = o$. Z indukčního předpokladu jsou $x_1, \ldots, x_{k-1}$ lineárně nezávislé, tedy $\alpha_1 = \ldots = \alpha_{k-1} = 0$. Dosazením zpět máme $\alpha_k x_k = o$, neboli $\alpha_k = 0$.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(Důsledek 10.33)</span></p>

Pokud matice $A \in \mathbb{C}^{n \times n}$ má $n$ navzájem různých vlastních čísel, pak je diagonalizovatelná.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Tvrzení 10.34)</span></p>

Buď $A, B \in \mathbb{C}^{n \times n}$. Pak matice $AB$ i $BA$ mají stejná vlastní čísla včetně násobností.

</div>

*Důkaz.* Matice $\begin{pmatrix} AB & 0 \\ B & 0 \end{pmatrix}$ resp. $\begin{pmatrix} 0 & 0 \\ B & BA \end{pmatrix}$ jsou blokově trojúhelníkové, proto mají stejná vlastní čísla jako $AB$ resp. $BA$, plus navíc $n$-násobné vlastní číslo 0. Tyto matice jsou podobné skrze matici $S = \begin{pmatrix} I & A \\ 0 & I \end{pmatrix}$. Předchozí věta platí i pro obdélníkové matice $A, B^T \in \mathbb{R}^{m \times n}$, ale znění platí pouze pro nenulová vlastní čísla; násobnost nulových vlastních čísel může být (a typicky je) odlišná.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 10.35 — Mocnina matice)</span></p>

Buď $A = S \Lambda S^{-1}$ spektrální rozklad matice $A \in \mathbb{C}^{n \times n}$. Pak $A^2 = S \Lambda S^{-1} S \Lambda S^{-1} = S \Lambda^2 S^{-1}$. Obecněji:

$$A^k = S \Lambda^k S^{-1} = S \begin{pmatrix} \lambda_1^k & 0 & 0 \\ 0 & \ddots & 0 \\ 0 & 0 & \lambda_n^k \end{pmatrix} S^{-1}.$$

Můžeme studovat i asymptotické chování. Zjednodušeně:

$$\lim_{k \to \infty} A^k = \begin{cases} 0, & \text{pokud } \rho(A) < 1, \\ \text{diverguje}, & \text{pokud } \rho(A) > 1, \\ \text{konverguje / diverguje}, & \text{pokud } \rho(A) = 1. \end{cases}$$

Geometricky: Mocnění matice $A$ odpovídá skládání zobrazení se sebou samým. Pokud jsou vlastní čísla v absolutní hodnotě menší než 1, tak lineární zobrazení kontrahuje vzdálenosti a proto konverguje k nule. Pokud je alespoň jedno vlastní číslo v absolutní hodnotě větší než 1, pak lineární zobrazení ve směru vlastního vektoru roztahuje vzdálenosti, a proto diverguje.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 10.36 — Rekurentní vzorečky a Fibonacci)</span></p>

Uvažujme posloupnost $a_1, a_2, \ldots$ zadanou rekurentním vztahem $a_n = p a_{n-1} + q a_{n-2}$, kde $a_1, a_2$ jsou dané první hodnoty posloupnosti a $p, q$ konstanty. Rekurenci vyjádříme maticově:

$$\begin{pmatrix} a_n \\ a_{n-1} \end{pmatrix} = \begin{pmatrix} p & q \\ 1 & 0 \end{pmatrix} \begin{pmatrix} a_{n-1} \\ a_{n-2} \end{pmatrix}.$$

Označíme-li $x_n \coloneqq \begin{pmatrix} a_n \\ a_{n-1} \end{pmatrix}$ a $A = \begin{pmatrix} p & q \\ 1 & 0 \end{pmatrix}$, pak rekurence má tvar $x_n = A x_{n-1} = A^2 x_{n-2} = \ldots = A^{n-2} x_2$.

Potřebujeme tedy určit vyšší mocninu matice $A$. K tomu poslouží diagonalizace: $A = S \Lambda S^{-1}$, a pak $x_n = S \Lambda^{n-2} S^{-1} x_2$. Explicitní vyjádření $a_n$ je skryto v první složce vektoru $x_n$.

Pro Fibonacciho posloupnost s $a_n = a_{n-1} + a_{n-2}$ a $a_1 = a_2 = 1$ a hodnotou zlatého řezu $\varphi \coloneqq \frac{1}{2}(1 + \sqrt{5})$ dostaneme

$$a_n = -\frac{\sqrt{5}}{5}(1 - \varphi)^n + \frac{\sqrt{5}}{5}\varphi^n.$$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 10.37 — Diskrétní a rychlá Fourierova transformace)</span></p>

Matice, které jsou ve tvaru jako má matice $A$ (tj. cirkulanty), mají pozoruhodné vlastnosti. Jedna z nich je, že její vlastní vektory nezávisí na konkrétních hodnotách $a_0, \ldots, a_{n-1}$, ale pouze na struktuře cirkulantu. Vlastní čísla matice $A$ se pak dají jednoduše dopočítat ze znalosti vlastních vektorů.

Pro matici $S$ navíc platí $S^{-1} = \frac{1}{n}\overline{S}^T$. Součin $Ab$ se dá nyní vyjádřit jako

$$Ab = S \Lambda \frac{1}{n} \overline{S}^T b.$$

Součin $Ab$ můžeme tedy vyjádřit pomocí tří operací: vynásobení postupně třemi maticemi $\frac{1}{n}\overline{S}^T$, $\Lambda$ a $S$. Pokud si matici $S$ představíme jako matici přechodu z báze vlastních vektorů do kanonické báze, pak první operace převede vektor $b$ do souřadného systému vlastních vektorů (tzv. *diskrétní Fourierova transformace*), druhá provede hlavní operaci a třetí převede výsledný vektor zpět do kanonické báze (tzv. *inverzní Fourierova transformace*). Druhá operace je zřejmě triviální, protože $\Lambda$ je diagonální.

Za použití vhodných algoritmů (založených např. na principu rozděl a panuj) lze vynásobení vektoru $b$ maticí $\frac{1}{n}\overline{S}^T$ provést v čase úměrný funkci $n \log(n)$. Podobně pro násobení maticí $S$. To je asymptoticky výrazné vylepšení oproti obyčejnému maticovému součinu $Ab$, který vyžaduje řádově $2n^2$ aritmetických operací. Takto vylepšená metoda se nazývá *rychlá Fourierova transformace* a je to jeden z nejvýznamnějších numerických algoritmů.

</div>

### Jordanova normální forma

Nejjednodušší tvar matice, ke kterému lze dospět pomocí elementárních řádkových úprav, je redukovaný odstupňovaný tvar. Jaký je však nejjednodušší tvar matice, ke kterému lze dospět pomocí podobnosti? Diagonální matice to není, protože již víme, že ne všechny matice jsou diagonalizovatelné. Nicméně každou matici lze podobnostní transformací převést na poměrně jednoduchý tvar, nazývaný Jordanova normální forma.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Definice 10.38 — Jordanova buňka)</span></p>

Buď $\lambda \in \mathbb{C}$, $k \in \mathbb{N}$. *Jordanova buňka* $J_k(\lambda)$ je čtvercová matice řádu $k$ definovaná

$$J_k(\lambda) = \begin{pmatrix} \lambda & 1 & 0 & \ldots & 0 \\ 0 & \ddots & \ddots & \ddots & \vdots \\ \vdots & \ddots & \ddots & \ddots & 0 \\ \vdots & & \ddots & \ddots & 1 \\ 0 & \ldots & \ldots & 0 & \lambda \end{pmatrix}.$$

Jordanova buňka má vlastní číslo $\lambda$, které je $k$-násobné, a přísluší mu pouze jeden vlastní vektor $e_1 = (1, 0, \ldots, 0)^T$, protože matice $J_k(\lambda) - \lambda I_k$ má hodnost $k - 1$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Definice 10.39 — Jordanova normální forma)</span></p>

Matice $J \in \mathbb{C}^{n \times n}$ je v *Jordanově normální formě*, pokud je v blokově diagonálním tvaru

$$J = \begin{pmatrix} J_{k_1}(\lambda_1) & 0 & \ldots & 0 \\ 0 & \ddots & \ddots & \vdots \\ \vdots & \ddots & \ddots & 0 \\ 0 & \ldots & 0 & J_{k_m}(\lambda_m) \end{pmatrix}$$

a na diagonále jsou Jordanovy buňky $J_{k_1}(\lambda_1), \ldots, J_{k_m}(\lambda_m)$.

</div>

Hodnoty $\lambda_i$ a $k_i$ nemusí být navzájem různé. Stejně tak nějaká Jordanova buňka se může vyskytovat vícekrát.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Věta 10.40 — O Jordanově normální formě)</span></p>

Každá matice $A \in \mathbb{C}^{n \times n}$ je podobná matici v Jordanově normální formě. Tato matice je až na pořadí buněk určena jednoznačně.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 10.41)</span></p>

Matice

$$A = \begin{pmatrix} 5 & -2 & 2 & -2 & 0 \\ 0 & 6 & -1 & 3 & 2 \\ 2 & 2 & 7 & -2 & -2 \\ 2 & 3 & 1 & 2 & -4 \\ -2 & -2 & -2 & 6 & 11 \end{pmatrix}$$

má vlastní číslo 5 (dvojnásobné) a 7 (trojnásobné). Protože $3 = \operatorname{rank}(A - 5I_5) = \operatorname{rank}(A - 5I_5)^2$, tak budeme hledat dva řetízky o délce 1. Najdeme dva lineárně nezávislé vektory $x_1, x_2 \in \operatorname{Ker}(A - 5I_5)$, například $x_1 = (-2, 1, 1, 0, 0)^T$ a $x_2 = (-1, 1, 0, -1, 1)^T$.

Přistupme k vlastnímu číslu 7. Nyní máme $\operatorname{rank}(A - 7I_5) = 3$ a $\operatorname{rank}(A - 7I_5)^2 = \operatorname{rank}(A - 7I_5)^3 = 2$. Zvolíme tedy $x_4 \in \operatorname{Ker}(A - 7I_5)^2 \setminus \operatorname{Ker}(A - 7I_5)$, například $x_4 = (1, 0, 1, 0, 0)^T$ a potom příslušnou část báze tvoří řetízek $x_3 = (A - 7I_5)x_4 = (0, -1, 2, 3, -4)^T$, $x_4$. Posledním vektorem báze bude vektor z $\operatorname{Ker}(A - 7I_5)$ lineárně nezávislý s vektorem $x_3$, tedy například $x_5 = (0, 1, 1, 0, 1)^T$.

Dáme-li tyto vektory $x_1, \ldots, x_5$ do sloupců matice $S$, pak

$$J = S^{-1}AS = \begin{pmatrix} 5 & 0 & 0 & 0 & 0 \\ 0 & 5 & 0 & 0 & 0 \\ 0 & 0 & 7 & 1 & 0 \\ 0 & 0 & 0 & 7 & 0 \\ 0 & 0 & 0 & 0 & 7 \end{pmatrix}$$

je hledaná Jordanova normální forma matice $A$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Tvrzení 10.42)</span></p>

Počet všech Jordanových buněk odpovídajících $\lambda$ je roven počtu vlastních vektorů pro $\lambda$.

</div>

Jako důsledek dále dostáváme, že (algebraická) násobnost každého vlastního čísla $\lambda$ je vždy větší nebo rovna počtu vlastních vektorů, které mu přísluší (tedy geometrické násobnosti).

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(Důsledek 10.43)</span></p>

Násobnost vlastního čísla je větší nebo rovna počtu vlastních vektorů, které mu přísluší.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Poznámka 10.44 — Velikosti a počet buněk)</span></p>

Počet buněk $J_k(\lambda)$ matice $A \in \mathbb{C}^{n \times n}$ ve výsledné Jordanově normální formě je roven

$$\operatorname{rank}(\bar{A}^{k-1}) - 2\operatorname{rank}(\bar{A}^k) + \operatorname{rank}(\bar{A}^{k+1}),$$

kde $\bar{A} = A - \lambda I_n$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Poznámka 10.45 — Zobecněné vlastní vektory)</span></p>

Buď $J$ Jordanova normální forma matice $A$, tedy $A = SJS^{-1}$ pro nějakou regulární matici $S$. Pokud je matice $J$ diagonální, pak ve sloupcích matice $S$ jsou vlastní vektory, které v pořadí odpovídají vlastním číslům matice $A$, umístěným na diagonále matice $J$. Sloupce matice $S$ se nazývají zobecněné vlastní vektory.

Souhrnně, buď $A \in \mathbb{C}^{n \times n}$ a $\lambda \in \mathbb{C}$ vlastní číslo. Prostor vlastních vektorů, příslušných k $\lambda$, tvoří jádro matice $A - \lambda I_n$. Prostor zobecněných vlastních vektorů, příslušných k $\lambda$, tvoří jádro matice $(A - \lambda I_n)^n$. Vlastních vektorů je plný počet (tedy $n$) pouze, když matice $A$ je diagonalizovatelná. Na druhou stranu, (lineárně nezávislých) zobecněných vlastních vektorů je vždy $n$. To přirozeně souvisí s tím, že každá matice je podobná matici v Jordanově normální formě.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 10.46 — Mocnění matice)</span></p>

Již v příkladu 10.35 jsme zmínili využití diagonalizace pro počítání mocniny matice. Pomocí Jordanovy normální formy můžeme tvrzení zobecnit pro libovolné $A \in \mathbb{C}^{n \times n}$: Buď $A = SJS^{-1}$, pak

$$A^k = SJ^kS^{-1} = S \begin{pmatrix} J_{k_1}(\lambda_1)^k & 0 & \ldots & 0 \\ 0 & \ddots & \ddots & \vdots \\ \vdots & \ddots & \ddots & 0 \\ 0 & \ldots & 0 & J_{k_m}(\lambda_m)^k \end{pmatrix} S^{-1}.$$

Asymptoticky pak dostaneme stejně jako pro diagonalizovatelné matice:

$$\lim_{k \to \infty} A^k = \begin{cases} 0, & \text{pokud } \rho(A) < 1, \\ \text{diverguje}, & \text{pokud } \rho(A) > 1, \\ \text{konverguje / diverguje}, & \text{pokud } \rho(A) = 1. \end{cases}$$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 10.47 — Maticová funkce)</span></p>

Položme si otázku: Jak zavést maticovou funkci jako např. $\cos(A)$, $e^A$ atp.? Pro reálnou funkci $f \colon \mathbb{R} \to \mathbb{R}$ a matici $A \in \mathbb{R}^{n \times n}$ lze zavést $f(A)$ tak, že aplikujeme funkci na každou složku matice zvlášť, ale takový přístup nemá moc pěkných vlastností.

Předpokládejme, že funkce $f \colon \mathbb{R} \to \mathbb{R}$ se dá vyjádřit nekonečným rozvojem $f(x) = \sum_{i=0}^{\infty} a_i x^i$; reálné analytické funkce jako např. $\sin(x)$, $\exp(x)$ aj. tento předpoklad splňují. Pak je tedy přirozené zavést $f(A) = \sum_{i=0}^{\infty} a_i A^i$. Mocnit matice již umíme, proto je-li $A = SJS^{-1}$, tak

$$f(A) = \sum_{i=0}^{\infty} a_i S J^i S^{-1} = S \left( \sum_{i=0}^{\infty} a_i J^i \right) S^{-1} = S f(J) S^{-1}.$$

Dále snadno nahlédneme, že

$$f(J) \coloneqq \begin{pmatrix} f(J_{k_1}(\lambda_1)) & 0 & \ldots & 0 \\ 0 & \ddots & \ddots & \vdots \\ \vdots & \ddots & \ddots & 0 \\ 0 & \ldots & 0 & f(J_{k_m}(\lambda_m)) \end{pmatrix}.$$

Pro $k_i = 1$ je to triviální, jde o matici řádu 1. Pro $k_i > 1$ je předpis složitější:

$$f(J_{k_i}(\lambda_i)) \coloneqq \begin{pmatrix} f(\lambda_i) & f'(\lambda_i) & \ldots & \frac{f^{(k_i - 1)}(\lambda_i)}{(k_i - 1)!} \\ 0 & \ddots & \ddots & \vdots \\ \vdots & \ddots & \ddots & f'(\lambda_i) \\ 0 & \ldots & 0 & f(\lambda_i) \end{pmatrix}.$$

Například funkce $f(x) = x^2$ má maticové rozšíření $f(A) = A^2$, jedná se tedy o klasické maticové mocnění. Jiným příkladem maticové funkce je maticová exponenciála $e^A = \sum_{i=0}^{\infty} \frac{1}{i!} A^i$. Jedno z mnoha využití je pro vyjádření rotací v prostoru $\mathbb{R}^3$. Matice $e^R$, kde

$$R = \alpha \begin{pmatrix} 0 & -z & y \\ z & 0 & -x \\ -y & x & 0 \end{pmatrix}$$

totiž popisuje matici rotace kolem osy se směrnicí $(x, y, z)^T$ o úhel $\alpha$ podle pravidla pravé ruky.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 10.48 — Soustava lineárních diferenciálních rovnic)</span></p>

Uvažme tzv. soustavu lineárních diferenciálních rovnic:

$$u(t)' = Au(t),$$

kde $A \in \mathbb{R}^{n \times n}$. Cílem je nalézt neznámou funkci $u \colon \mathbb{R} \to \mathbb{R}^n$ splňující tuto soustavu pro určitou počáteční podmínku tvaru $u(t_0) = u_0$.

Pro případ $n = 1$ je řešením diferenciální rovnice $u(t)' = au(t)$ funkce $u(t) = v \cdot e^{at}$, kde $v \in \mathbb{R}^n$ je libovolné. To nás motivuje hledat řešení obecného případu ve tvaru $u(t) = (v_1 e^{\lambda_1 t}, \ldots, v_n e^{\lambda_n t}) = e^{\lambda t} v$, kde $v_i, \lambda$ jsou neznámé. Dosazením $u(t) \coloneqq e^{\lambda t} v$ do soustavy dostaneme $\lambda e^{\lambda t} v = e^{\lambda t} Av$, neboli $\lambda v = Av$.

To je přímo úloha výpočtu vlastních čísel a vektorů. Nechť matice $A$ má vlastní čísla $\lambda_1, \ldots, \lambda_n$ a vlastní vektory $x_1, \ldots, x_n$. Pak řešení je $u(t) = \sum_{i=1}^n \alpha_i e^{\lambda_i t} x_i$, kde $\alpha_i \in \mathbb{R}$ se získá z počátečních podmínek.

Uvažme konkrétní příklad: $u_1'(t) = 7u_1(t) - 4u_2(t)$, $u_2'(t) = 5u_1(t) - 2u_2(t)$. Matice $A = \begin{pmatrix} 7 & -4 \\ 5 & -2 \end{pmatrix}$ má vlastní čísla 2 a 3, jim odpovídají vlastní vektory $(4, 5)^T$ a $(1, 1)^T$. Řešení úlohy jsou tvaru

$$\begin{pmatrix} u_1(t) \\ u_2(t) \end{pmatrix} = \alpha_1 e^{2t} \begin{pmatrix} 4 \\ 5 \end{pmatrix} + \alpha_2 e^{3t} \begin{pmatrix} 1 \\ 1 \end{pmatrix}, \quad \alpha_1, \alpha_2 \in \mathbb{R}.$$

Vlastní čísla také určují, jak se řešení $u(t)$ chová v delším čase. Pokud jsou vlastní čísla záporná, $e^{\lambda_i t}$ konverguje k nule pro $t \to \infty$. V tomto případě je úloha tzv. asymptoticky stabilní. Úloha je nestabilní, pokud nějaké vlastní číslo je kladné, protože pak $e^{\lambda_i t}$ diverguje pro $t \to \infty$.

</div>

### Symetrické matice

Reálné symetrické matice mají řadu pozoruhodných vlastností týkajících se vlastních čísel. Mezi stěžejní vlastnosti patří to, že jsou vždy diagonalizovatelné, jejich vlastní čísla jsou reálná a vlastní vektory jdou vybrat tak, aby byly na sebe kolmé.

Nejprve se podívejme na zobecnění transpozice a symetrických matic pro komplexní matice.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Definice 10.49 — Hermitovská matice a transpozice)</span></p>

*Hermitovská transpozice* matice $A \in \mathbb{C}^{n \times n}$ je matice $A^* \coloneqq \overline{A}^T$. Matice $A \in \mathbb{C}^{n \times n}$ se nazývá *hermitovská*, pokud $A^* = A$.

</div>

Hermitovská transpozice má podobné vlastnosti jako klasická transpozice, např. $(A^*)^* = A$, $(\alpha A)^* = \overline{\alpha} A^*$, $(A + B)^* = A^* + B^*$, $(AB)^* = B^* A^*$.

Pomocí hermitovské transpozice můžeme *unitární matice* (rozšiřující pojem ortogonální matice pro komplexní matice) definovat jako matice $Q \in \mathbb{C}^{n \times n}$ splňující $Q^* Q = I_n$.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 10.50)</span></p>

Z matic $\begin{pmatrix} 2 & 1 + i \\ 1 + i & 5 \end{pmatrix}$ a $\begin{pmatrix} 2 & 1 + i \\ 1 - i & 5 \end{pmatrix}$ je první symetrická, ale ne hermitovská, a druhá je hermitovská, ale ne symetrická. Pro reálné matice oba pojmy splývají.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Věta 10.51 — Vlastní čísla symetrických matic)</span></p>

Vlastní čísla reálných symetrických (resp. obecněji komplexních hermitovských) matic jsou reálná.

</div>

*Důkaz.* Buď $A \in \mathbb{C}^{n \times n}$ hermitovská a buď $\lambda \in \mathbb{C}$ její libovolné vlastní číslo a $x \in \mathbb{C}^n$ příslušný vlastní vektor jednotkové velikosti, tj. $\|x\|_2 = 1$. Přenásobením rovnice $Ax = \lambda x$ vektorem $x^*$ máme $x^* Ax = \lambda x^* x = \lambda$. Nyní

$$\lambda = x^* Ax = x^* A^* x = (x^* Ax)^* = \lambda^*.$$

Tedy $\lambda = \lambda^*$, a proto musí být $\lambda$ reálné.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 10.52 — Vlastní čísla matice projekce)</span></p>

Buď $P \in \mathbb{R}^{n \times n}$ matice projekce do podprostoru $U$ dimenze $d$. Pro každý vektor $x \in U$ platí $Px = x$. Tudíž 1 je vlastním číslem a odpovídá mu $d$ vlastních vektorů z báze prostoru $U$. Pro každý vektor $x \in U^\perp$ platí $Px = o$. Tedy 0 je vlastním číslem a odpovídá mu $n - d$ vlastních vektorů z báze prostoru $U^\perp$. Jiná vlastní čísla už matice $P$ nemá, protože jsme našli $n$ lineárně nezávislých vlastních vektorů. Souhrnně, matice projekce má vlastní čísla pouze 0 a 1.

</div>

Následující věta říká, že symetrické matice jsou diagonalizovatelné. Navíc jsou diagonalizovatelné specifickým způsobem: z vlastních vektorů lze vybrat ortonormální systém, tedy matice podobnosti je ortogonální.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Věta 10.53 — Spektrální rozklad symetrických matic)</span></p>

Pro každou symetrickou matici $A \in \mathbb{R}^{n \times n}$ existuje ortogonální $Q \in \mathbb{R}^{n \times n}$ a diagonální $\Lambda \in \mathbb{R}^{n \times n}$ takové, že $A = Q \Lambda Q^T$.

</div>

*Důkaz.* Matematickou indukcí podle $n$. Případ $n = 1$ je triviální: $\Lambda = A$, $Q = 1$. Indukční krok $n \leftarrow n - 1$. Buď $\lambda$ vlastní číslo $A$ a $x$ odpovídající vlastní vektor normovaný $\|x\|_2 = 1$. Doplňme $x$, jakožto ortonormální systém, na ortogonální matici $S \coloneqq (x \mid \cdots)$. Protože $(A - \lambda I_n)x = o$, máme $(A - \lambda I_n)S = (o \mid \cdots)$, a tudíž $S^T(A - \lambda I_n)S = S^T(o \mid \cdots) = (o \mid \cdots)$. A jelikož je tato matice symetrická, máme

$$S^T(A - \lambda I_n)S = \begin{pmatrix} 0 & o^T \\ o & A' \end{pmatrix},$$

kde $A'$ je nějaká symetrická matice řádu $n - 1$. Podle indukčního předpokladu má spektrální rozklad $A' = Q' \Lambda' Q'^T$, kde $\Lambda'$ je diagonální a $Q'$ ortogonální. Matice a rovnost rozšíříme o jeden řad a nakonec dostaneme hledaný rozklad $A = Q \Lambda Q^T$, kde $Q \coloneqq SR$ je ortogonální matice a $\Lambda \coloneqq \Lambda'' + \lambda I_n$ je diagonální.

Podobně můžeme spektrálně rozložit hermitovské matice $A = Q \Lambda Q^*$, kde $Q$ je unitární matice.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Poznámka 10.54 — Jiná forma spektrálního rozkladu)</span></p>

Nechť symetrická $A \in \mathbb{R}^{n \times n}$ má vlastní čísla $\lambda_1, \ldots, \lambda_n$ a odpovídající ortonormální vlastní vektory $x_1, \ldots, x_n$. Tedy ve spektrálním rozkladu $A = Q \Lambda Q^T$ je $\Lambda_{ii} = \lambda_i$ a $Q_{*i} = x_i$. Pokud rozepíšeme $\Lambda$ na součet jednodušších diagonálních matic

$$\Lambda = \sum_{i=1}^n \lambda_i e_i e_i^T,$$

tak matici $A$ lze vyjádřit jako

$$A = Q \Lambda Q^T = Q \left( \sum_{i=1}^n \lambda_i e_i e_i^T \right) Q^T = \sum_{i=1}^n \lambda_i Q e_i e_i^T Q^T = \sum_{i=1}^n \lambda_i Q_{*i} Q_{*i}^T = \sum_{i=1}^n \lambda_i x_i x_i^T.$$

Tvar $A = \sum_{i=1}^n \lambda_i x_i x_i^T$ je tak alternativní vyjádření spektrálního rozkladu, ve kterém matici $A$ rozepisujeme na součet $n$ matic hodnosti 0 nebo 1. Navíc, $x_i x_i^T$ je matice projekce na přímku $\operatorname{span}\lbrace x_i \rbrace$, tudíž z geometrického hlediska se na zobrazení $x \mapsto Ax$ můžeme dívat jako součet $n$ zobrazení, kde v každém provádíme projekci na přímku (kolmou na ostatní) a škálování podle hodnoty $\lambda_i$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Věta 10.55 — Courant–Fischer)</span></p>

Nechť $\lambda_1 \ge \cdots \ge \lambda_n$ jsou vlastní čísla symetrické matice $A \in \mathbb{R}^{n \times n}$. Pak

$$\lambda_1 = \max_{x: \|x\|_2 = 1} x^T A x, \quad \lambda_n = \min_{x: \|x\|_2 = 1} x^T A x.$$

</div>

*Důkaz.* Pouze pro $\lambda_1$, druhá část je analogická. Nerovnost „$\le$": Buď $x_1$ vlastní vektor příslušný k $\lambda_1$ normovaný $\|x_1\|_2 = 1$. Pak $Ax_1 = \lambda_1 x_1$. Přenásobením $x_1^T$ zleva dostaneme $\lambda_1 = \lambda_1 x_1^T x_1 = x_1^T A x_1 \le \max_{x: \|x\|_2 = 1} x^T Ax$. Nerovnost „$\ge$": Buď $x \in \mathbb{R}^n$ libovolný vektor takový, že $\|x\|_2 = 1$. Označme $y \coloneqq Q^T x$, pak $\|y\|_2 = 1$. S využitím spektrálního rozkladu $A = Q\Lambda Q^T$ dostaneme

$$x^T Ax = x^T Q \Lambda Q^T x = y^T \Lambda y = \sum_{i=1}^n \lambda_i y_i^2 \le \sum_{i=1}^n \lambda_1 y_i^2 = \lambda_1 \|y\|_2^2 = \lambda_1.$$

### Teorie nezáporných matic

Perronova–Frobeniova teorie nezáporných matic je pokročilá teorie kolem vlastních čísel nezáporných matic. Uvedeme jen základní Perronův výsledek bez důkazu. Matice $A \in \mathbb{R}^{n \times n}$ se nazývá *nezáporná*, pokud je nezáporná v každé složce ($a_{ij} \ge 0$ pro všechna $i, j$), a nazývá se *kladná*, pokud je kladná v každé složce ($a_{ij} > 0$ pro všechna $i, j$).

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Věta 10.56 — Perronova)</span></p>

1. Buď $A \in \mathbb{R}^{n \times n}$ nezáporná matice. Pak v absolutní hodnotě největší vlastní číslo je reálné nezáporné a příslušný vlastní vektor je nezáporný (ve všech složkách).
2. Buď $A \in \mathbb{R}^{n \times n}$ kladná matice. Pak v absolutní hodnotě největší vlastní číslo je reálné kladné, je jediné (ostatní mají menší absolutní hodnotu), má násobnost 1, a příslušný vlastní vektor je kladný (ve všech složkách). Navíc žádnému jinému vlastnímu číslu neodpovídá nezáporný vlastní vektor.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 10.57 — Markovovy řetězce)</span></p>

Jedním z využití mocnin matice (příklad 10.46) a trochu i teorie nezáporných matic jsou Markovovy řetězce. Buď $x \in \mathbb{R}^n$ stavový vektor, čili $x_i$ udává hodnotu stavu $i$. Buď $A \in \mathbb{R}^{n \times n}$ matice s hodnotami $a_{ij} \in [0, 1]$ takovými, že součet hodnot v každém sloupci je roven 1. Na matici $A$ budeme nahlížet jako na přechodovou matici, to jest, hodnota $a_{ij}$ je pravděpodobnost přechodu ze stavu $j$ do stavu $i$. Pak $Ax$ udává nový stavový vektor po jednom kroku daného procesu. Zajímá nás, jak se bude stavový vektor vyvíjet v čase a zda se nějak ustálí.

Přímo z definice platí, že $A^T e = e$, kde $e = (1, \ldots, 1)^T$. Tudíž $e$ je vlastní vektor $A^T$ a 1 vlastní číslo $A^T$ (a tedy i $A$). Navíc se dá ukázat, že žádné jiné vlastní číslo není v absolutní hodnotě větší (viz poznámka 10.60), tudíž 1 je vlastní číslo $A$ z Perronovy věty a odpovídající vlastní vektor je nezáporný.

Konkrétní příklad: Migrace obyvatel USA v sektorech město–předměstí–venkov probíhá každoročně podle vzorce:

| | z města | z předměstí | z venkova |
|---|---|---|---|
| zůstane ve městě | 96% | 1% | 1.5% |
| do předměstí | 3% | 98% | 0.5% |
| na venkov | 1% | 1% | 98% |

Počáteční stav: 58 mil. obyvatel ve městě, 142 mil. na předměstí a 60 mil. na venkově. Označme

$$A \coloneqq \begin{pmatrix} 0.96 & 0.01 & 0.015 \\ 0.03 & 0.98 & 0.005 \\ 0.01 & 0.01 & 0.98 \end{pmatrix}, \quad x_0 = (58, 142, 60)^T.$$

Diagonalizací spočítáme $A^\infty x_0 = (0.23 e^T x_0, \; 0.43 e^T x_0, \; 0.33 e^T x_0)^T$. Tedy (bez ohledu na počáteční stav $x_0$) se rozložení obyvatelstva ustálí na hodnotách: 23% ve městě, 43% předměstí, 33% venkov.

</div>

### Výpočet vlastních čísel

Jak jsme již zmínili, vlastní čísla se počítají pouze numerickými iteračními metodami a hledat je jako kořeny charakteristického polynomu není efektivní postup. V této sekci ukážeme jednoduchý odhad na vlastní čísla a jednoduchou metodu na výpočet největšího vlastního čísla.

Protože numerické metody jsou iterativní a počítají vlastní čísla jenom s určitou přesností, je těžké vyjádřit a priori přesně počet operací, které vykonají. Nicméně, současné metody pro symetrické i nesymetrické matice mají prakticky kubickou složitost. To znamená, že vyžadují asymptoticky $\alpha n^3$ operací, kde $n$ je rozměr matice a $\alpha > 0$ příslušný koeficient.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Věta 10.58 — Gerschgorinovy disky)</span></p>

Každé vlastní číslo $\lambda$ matice $A \in \mathbb{C}^{n \times n}$ leží v kruhu o středu $a_{ii}$ a poloměru $\sum_{j \neq i} |a_{ij}|$ pro nějaké $i \in \lbrace 1, \ldots, n \rbrace$.

</div>

*Důkaz.* Buď $\lambda$ vlastní číslo a $x$ odpovídající vlastní vektor, tedy $Ax = \lambda x$. Nechť $i$-tá složka $x$ má největší absolutní hodnotu, tj. $|x_i| = \max_{k=1,\ldots,n} |x_k|$. Protože $i$-tá rovnice má tvar $\sum_{j=1}^n a_{ij} x_j = \lambda x_i$, vydělením $x_i \neq 0$ dostáváme $\lambda = a_{ii} + \sum_{j \neq i} a_{ij} \frac{x_j}{x_i}$, a tím pádem $|\lambda - a_{ii}| = \left| \sum_{j \neq i} a_{ij} \frac{x_j}{x_i} \right| \le \sum_{j \neq i} |a_{ij}| \frac{|x_j|}{|x_i|} \le \sum_{j \neq i} |a_{ij}|$.

Věta dává jednoduchý ale hrubý odhad na velikost vlastních čísel (existují i vylepšení, např. Cassiniho ovály aj.). Nicméně, v některých aplikacích může takovýto odhad postačovat.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 10.59)</span></p>

Mějme $A = \begin{pmatrix} 2 & 1 & 0 \\ -2 & 5 & 1 \\ -1 & -2 & -3 \end{pmatrix}$. Vlastní čísla matice $A$ jsou $\lambda_1 = -2.78$, $\lambda_2 = 3.39 + 0.6i$, $\lambda_3 = 3.39 - 0.6i$. Gerschgorinovy disky jsou: kruh se středem 2 a poloměrem 1, kruh se středem 5 a poloměrem 3, kruh se středem $-3$ a poloměrem 3. Všechna vlastní čísla leží uvnitř některého z disků.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Poznámka 10.60 — Tři použití Gerschgorinových disků)</span></p>

1. *Kriterium pro zastavení výpočtu iteračních metod.* Například Jacobiho metoda na výpočet vlastních čísel spočívá v postupném zmenšování nediagonálních prvků symetrické matice, takže matice konverguje k diagonální matici. Gerschgorinovy disky pak dávají horní mez na přesnost vypočtených vlastních čísel. Pokud např. matice $A \in \mathbb{R}^{n \times n}$ je skoro diagonální v tom smyslu, že všechny mimodiagonální prvky jsou menší než $10^{-k}$ pro nějaké $k \in \mathbb{N}$, pak diagonální prvky aproximují vlastní čísla s přesností $10^{-k}(n-1)$.
2. *Diagonálně dominantní matice.* Gerschgorinovy disky dávají také následující postačující podmínku pro regularitu matice $A \in \mathbb{C}^{n \times n}$: $|a_{ii}| > \sum_{j \neq i} |a_{ij}|$ $\forall i = 1, \ldots, n$. V tomto případě totiž disky neobsahují počátek a proto nula není vlastním číslem $A$. Matice s touto vlastností se nazývají diagonálně dominantní.
3. *Markovovy matice.* Buď $A$ Markovova matice z příkladu 10.57. Všechny Gerschgorinovy disky matice $A^T$ mají střed v bodě v intervalu $[0, 1]$ a pravým krajem protínají hodnotu 1 na reálné ose. To dokazuje, že $\rho(A) \le 1$, a tudíž číslo 1 je skutečně v absolutní hodnotě největším vlastním číslem matice $A$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Algoritmus 10.61 — Mocninná metoda)</span></p>

Vstup: matice $A \in \mathbb{C}^{n \times n}$.

1. Zvol $o \neq x_0 \in \mathbb{C}^n$, $i \coloneqq 1$,
2. **while not** splněna ukončovací podmínka **do**
3. &emsp; $y_i \coloneqq A x_{i-1}$,
4. &emsp; $x_i \coloneqq \frac{1}{\|y_i\|_2} y_i$,
5. &emsp; $i \coloneqq i + 1$,
6. **end while**

Výstup: $\lambda_1 \coloneqq x_{i-1}^T y_i$ je odhad vlastního čísla, $v_1 \coloneqq x_i$ je odhad příslušného vlastního vektoru.

</div>

Metodu ukončíme ve chvíli, kdy se hodnota $x_{i-1}^T y_i$ resp. vektor $x_i$ ustálí; potom $x_i \approx x_{i-1}$ je odhad vlastního vektoru a $x_{i-1}^T y_i = x_{i-1}^T A x_{i-1} \approx x_{i-1}^T \lambda x_{i-1} \approx \lambda$ odhad odpovídajícího vlastního čísla. Metoda může být pomalá, špatně se odhaduje chyba a míra konvergence a navíc velmi záleží na počáteční volbě $x_0$. Na druhou stranu je robustní (zaokrouhlovací chyby nemají velký vliv) a snadno aplikovatelná na velké řídké matice. Ne vždy konverguje, ale za určitých předpokladů se to dá zajistit.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Tvrzení 10.63 — Konvergence mocninné metody)</span></p>

Buď $A \in \mathbb{R}^{n \times n}$ s vlastními čísly $|\lambda_1| > |\lambda_2| \ge \ldots \ge |\lambda_n|$ a odpovídajícími lineárně nezávislými vlastními vektory $v_1, \ldots, v_n$ velikosti 1. Nechť $x_0$ má nenulovou souřadnici ve směru $v_1$. Pak $x_i$ konverguje (až na násobek) k vlastnímu vektoru $v_1$ a $x_{i-1}^T y_i$ konverguje k vlastnímu číslu $\lambda_1$.

</div>

*Důkaz.* Protože vektory $v_1, \ldots, v_n$ tvoří bázi prostoru $\mathbb{R}^n$, můžeme vektor $x_0$ vyjádřit jako $x_0 = \sum_{j=1}^n \alpha_j v_j$, kde $\alpha_1 \neq 0$ podle předpokladu. Pak $A^i x_0 = \sum_{j=1}^n \alpha_j \lambda_j^i v_j = \lambda_1^i \left( \alpha_1 v_1 + \sum_{j \neq 1} \alpha_j \left(\frac{\lambda_j}{\lambda_1}\right)^i v_j \right)$. Protože vektory $x_i$ postupně normujeme, násobek $\lambda_1^i$ nás nemusí zajímat. Zbylý vektor postupně konverguje k $\alpha_1 v_1$, protože $\left|\frac{\lambda_j}{\lambda_1}\right| < 1$ a tudíž $\left|\frac{\lambda_j}{\lambda_1}\right|^i \to 0$ pro $i \to \infty$.

Z důkazu věty vidíme, že rychlost konvergence výrazně závisí na poměru $\left|\frac{\lambda_2}{\lambda_1}\right|$.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Tvrzení 10.64 — O deflaci vlastního čísla)</span></p>

Buď $A \in \mathbb{R}^{n \times n}$ symetrická, $\lambda_1, \ldots, \lambda_n$ její vlastní čísla a $v_1, \ldots, v_n$ odpovídající ortonormální vlastní vektory. Pak matice $A - \lambda_1 v_1 v_1^T$ má vlastní čísla $0, \lambda_2, \ldots, \lambda_n$ a vlastní vektory $v_1, \ldots, v_n$.

</div>

*Důkaz.* Podle poznámky 10.54 lze psát $A = \sum_{i=1}^n \lambda_i v_i v_i^T$. Pak $A - \lambda_1 v_1 v_1^T = 0 v_1 v_1^T + \sum_{i=2}^n \lambda_i v_i v_i^T$, což je spektrální rozklad matice $A - \lambda_1 v_1 v_1^T$.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Poznámka 10.65 — K deflaci vlastního čísla obecné matice)</span></p>

Buď $\lambda$ vlastní číslo a $x$ odpovídající vlastní vektor matice $A \in \mathbb{R}^{n \times n}$. Doplňme $x$ na regulární matici $S$ tak, aby $S_{*1} = x$. Pak

$$S^{-1}AS = S^{-1}A(x \mid \cdots) = S^{-1}(\lambda x \mid \cdots) = (\lambda e_1 \mid \cdots) = \begin{pmatrix} \lambda & \cdots \\ o & A' \end{pmatrix}.$$

Z podobnosti má matice $A'$ stejná vlastní čísla jako $A$, pouze $\lambda$ má o jedna menší násobnost. Tudíž zbývající vlastní čísla matice $A$ můžeme najít pomocí $A'$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 10.66 — Vyhledávač Google a PageRank)</span></p>

Uvažujme webovou síť s $N$ webovými stránkami. Cílem je stanovit důležitosti $x_1, \ldots, x_N$ jednotlivých stránek. Základní myšlenka autorů googlovského PageRanku spočívá ve stanovení důležitosti $i$-té stránky tak, aby byla úměrná součtu důležitosti stránek na ni odkazujících. Řešíme tedy rovnici $x_i = \sum_{j=1}^N \frac{a_{ij}}{b_j} x_j$, $i = 1, \ldots, N$, kde $a_{ij} = 1$ pokud $j$-tá stránka odkazuje na $i$-tou (jinak 0) a $b_j$ je počet odkazů z $j$-té stránky. Maticově $A'x = x$, kde $a'_{ij} \coloneqq \frac{a_{ij}}{b_j}$.

Tedy $x$ je vlastní vektor matice $A'$ příslušný vlastnímu číslu 1. Vlastní číslo 1 je dominantní, což snadno nahlédneme z Gerschgorinových disků pro matici $A'^T$ (součet sloupců matice $A'$ je roven 1, tedy všechny Gerschgorinovy disky mají nejpravější konec v bodě 1). Podle Perronovy věty 10.56 je vlastní vektor $x$ nezáporný.

V praxi je matice $A'$ obrovská, řádově $\approx 10^{10}$, a zároveň řídká (většina hodnot jsou nuly). Proto se na výpočet $x$ hodí mocninná metoda, stačí $\approx 100$ iterací. Prakticky se navíc matice $A'$ trochu upravuje, aby byla stochastická, aperiodická a ireducibilní.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Poznámka 10.67 — Další aplikace v teorii grafů)</span></p>

Na závěr zmiňme široké použití vlastních čísel v teorii grafů. Vlastní čísla matice sousednosti a Laplaceovy matice grafu říkají mnoho o tom, jaká je struktura grafu. Používají se k odhadování velikosti tzv. „úzkého hrdla" v grafu, což je množina vrcholů s relativně málo hranami vedoucími ven. Dávají také různé odhady na velikost nezávislé množiny v grafu a jiné charakteristiky.

</div>

### Shrnutí ke kapitole 10

Vlastní čísla a vlastní vektory matice $A$ poskytují o matici a o lineárním zobrazení $x \mapsto Ax$ podstatné informace. Geometricky vlastní vektory reprezentují invariantní směry, které se zobrazí samy na sebe, a vlastní čísla představují míru škálování v těchto směrech. Vlastní čísla tedy celkem dobře popisují jak moc lineární zobrazení $x \mapsto Ax$ degeneruje objekty a co se děje, když zobrazení iterujeme.

Vlastních čísel matice $A \in \mathbb{C}^{n \times n}$ je právě $n$ včetně násobností, a (lineárně nezávislých) vlastních vektorů je nanejvýš $n$. Má-li vlastní číslo násobnost $k$, pak mu odpovídá maximálně $k$ vlastních vektorů. Různá vlastní čísla pak mají lineárně nezávislé vlastní vektory.

Elementární úpravy mění vlastní čísla matice, ale úprava, která je nemění, je podobnost. Geometricky podobnost znamená jen změnu souřadného systému. Každá matice je podobná matici v jednodušším tvaru — Jordanově normálním tvaru. Ten může mít dokonce podobu diagonální matice (pak je původní matice diagonalizovatelná). To nastává právě tehdy, když matice má plný počet vlastních vektorů (počet Jordanových buněk je roven počtu vlastních vektorů).

Důležitou třídou matic jsou symetrické matice. Mají tři podstatné vlastnosti: (1) jsou vždy diagonalizovatelné, (2) mají reálná vlastní čísla, (3) vlastní vektory jdou vybrat tak, aby na sebe byly kolmé. Příslušný spektrální rozklad je pak velmi užitečný nástroj.

Další třídou matic se speciálními vlastnostmi jsou nezáporné matice: největší vlastní číslo v absolutní hodnotě leží na reálné ose vpravo od počátku a odpovídající vlastní vektor je nezáporný.

Problém výpočtu vlastních čísel a výpočtu kořenů polynomu jsou na sebe převoditelné (skrze charakteristický polynom a matici společnici). Na výpočet vlastních čísel nelze jednoduše použít Gaussovu eliminaci — veškeré používané metody jsou iterativní, jako například mocninná metoda. Šikovné jsou i různé odhady jako jsou například Gerschgorinovy disky.

## Kapitola 11 — Positivně (semi-)definitní matice

Již ve větě 10.55 jsme se setkali s funkcí $f \colon \mathbb{R}^n \to \mathbb{R}$ danou předpisem $f(x) = x^T Ax = \sum_{i=1}^n \sum_{j=1}^n a_{ij} x_i x_j$, kde $A \in \mathbb{R}^{n \times n}$ je pevná matice. Tato funkce představuje polynom v proměnných $x_1, \ldots, x_n$ a více ji budeme rozebírat v kapitole 12. Zde se zaměříme na situaci, kdy funkce $f(x)$ je nezáporná resp. kladná a pro jaké matice je to splněno.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Definice 11.1 — Positivně (semi-)definitní matice)</span></p>

Buď $A \in \mathbb{R}^{n \times n}$ symetrická. Pak $A$ je *positivně semidefinitní*, pokud $x^T Ax \ge 0$ pro všechna $x \in \mathbb{R}^n$, a $A$ je *positivně definitní*, pokud $x^T Ax > 0$ pro všechna $x \neq o$.

</div>

Zřejmě, je-li $A$ positivně definitní, pak je i positivně semidefinitní.

Positivní definitnost a semidefinitnost není potřeba testovat pro všechny vektory $x \in \mathbb{R}^n$, ale stačí se omezit například na jednotkovou sféru. Pokud platí $x^T Ax > 0$ pro všechny vektory $x$ s jednotkovou normou $\|x\|_2 = 1$, pak to platí i pro ostatní nenulové vektory. Každý vektor $x \neq o$ je totiž kladným násobkem vektoru jednotkové délky, konkrétně $\|x\|_2$-násobkem vektoru $\frac{1}{\|x\|_2} x$.

Kromě positivně (semi-)definitních matic lze zavést i negativně (semi-)definitní matice pomocí obrácené nerovnosti. Zabývat se jimi nebudeme, protože $A$ je negativně (semi-)definitní právě tehdy, když $-A$ je positivně (semi-)definitní, čili vše se redukuje na základní případ.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Poznámka 11.2)</span></p>

Definice dává smysl i pro nesymetrické matice, ale ty můžeme snadno zsymetrizovat úpravou $\frac{1}{2}(A + A^T)$, neboť

$$x^T \tfrac{1}{2}(A + A^T)x = \tfrac{1}{2}x^T Ax + \tfrac{1}{2}x^T A^T x = \tfrac{1}{2}x^T Ax + \left(\tfrac{1}{2}x^T Ax\right)^T = x^T Ax.$$

Tedy pro testování podmínky lze ekvivalentně použít symetrickou matici $\frac{1}{2}(A + A^T)$. Omezení na symetrické matice je tudíž bez újmy na obecnosti. Důvod, proč se omezujeme na symetrické matice, je, že řada testovacích podmínek funguje pouze pro symetrické matice.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 11.3)</span></p>

Příkladem positivně semidefinitní matice je $0_n$. Příkladem positivně definitní matice je $I_n$, neboť $x^T I_n x = x^T x = \|x\|_2^2 > 0$ pro všechna $x \neq o$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Poznámka 11.4 — Nutná podmínka pro positivní (semi-)definitnost)</span></p>

Buď $A \in \mathbb{R}^{n \times n}$ symetrická matice. Aby byla positivně semidefinitní, musí podle definice $x^T Ax \ge 0$ pro všechna $x \in \mathbb{R}^n$. Postupným dosazením $x = e_i$, $i = 1, \ldots, n$, dostaneme $x^T Ax = e_i^T A e_i = a_{ii} \ge 0$. Tím pádem, positivně semidefinitní matice musí mít nezápornou diagonálu a positivně definitní matice dokonce kladnou.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Poznámka 11.5)</span></p>

Matice $A = (a) \in \mathbb{R}^{1 \times 1}$ je positivně semidefinitní právě tehdy, když $a \ge 0$, a positivně definitní právě tehdy, když $a > 0$. Tím pádem se můžeme dívat na positivní semidefinitnost jako na zobecnění pojmu nezápornosti z čísel na matice. Proto se také positivní semidefinitnost matice $A \in \mathbb{R}^{n \times n}$ značívá $A \succeq 0$ (narozdíl od $A \ge 0$, což se používá pro nezápornost v každé složce).

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Tvrzení 11.6 — Vlastnosti positivně definitních matic)</span></p>

1. Jsou-li $A, B \in \mathbb{R}^{n \times n}$ positivně definitní, pak i $A + B$ je positivně definitní,
2. Je-li $A \in \mathbb{R}^{n \times n}$ positivně definitní a $\alpha > 0$, pak i $\alpha A$ je positivně definitní,
3. Je-li $A \in \mathbb{R}^{n \times n}$ positivně definitní, pak je regulární a $A^{-1}$ je positivně definitní.

</div>

*Důkaz.* První dvě vlastnosti jsou triviální, dokážeme pouze tu třetí. Nejprve ověříme regularitu matice $A$. Buď $x$ řešení soustavy $Ax = o$. Pak $x^T Ax = x^T o = 0$. Z předpokladu musí $x = o$. Nyní ukážeme positivní definitnost. Sporem nechť existuje $x \neq o$ takové, že $x^T A^{-1}x \le 0$. Pak $x^T A^{-1}x = x^T A^{-1}AA^{-1}x = y^T Ay \le 0$, kde $y = A^{-1}x \neq o$. To je spor, neboť $A$ je positivně definitní.

Analogie věty platí i pro positivně semidefinitní matice. Část (1) platí beze změny, část (2) platí pro všechna $\alpha \ge 0$, ale část (3) už obecně neplatí, protože positivně semidefinitní matice může být singulární.

Součinem positivně definitních matic se zabýváme v poznámce 12.20; s tímto výrazem jsme se skrytě setkali už například v metodě nejmenších čtverců (sekce 8.5) u soustavy normálních rovnic $A^T Ax = A^T b$ nebo obecně při explicitním vyjádření ortogonální projekce v $\mathbb{R}^n$ (věta 8.49).

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Věta 11.7 — Charakterizace positivní definitnosti)</span></p>

Buď $A \in \mathbb{R}^{n \times n}$ symetrická. Pak následující podmínky jsou ekvivalentní:

1. $A$ je positivně definitní,
2. vlastní čísla $A$ jsou kladná,
3. existuje matice $U \in \mathbb{R}^{m \times n}$ hodnosti $n$ taková, že $A = U^T U$.

</div>

*Důkaz.* Implikace (1) $\Rightarrow$ (2): Sporem nechť existuje vlastní číslo $\lambda \le 0$, a $x$ je příslušný vlastní vektor s eukleidovskou normou rovnou 1. Pak $Ax = \lambda x$ implikuje $x^T Ax = \lambda x^T x = \lambda \le 0$. To je spor s positivní definitností $A$.

Implikace (2) $\Rightarrow$ (3): Protože $A$ je symetrická, má spektrální rozklad $A = Q\Lambda Q^T$, kde $\Lambda$ je diagonální matice s prvky $\lambda_1, \ldots, \lambda_n > 0$. Definujme matici $\Lambda'$ jako diagonální s prvky $\sqrt{\lambda_1}, \ldots, \sqrt{\lambda_n} > 0$. Pak hledaná matice je například $U = \Lambda' Q^T$, neboť $U^T U = Q\Lambda' \Lambda' Q^T = Q\Lambda'^2 Q^T = Q\Lambda Q^T = A$. Uvědomme si, že $U$ má hodnost $n$ a je tudíž regulární, neboť je součinem dvou regulárních matic.

Implikace (3) $\Rightarrow$ (1): Sporem nechť $x^T Ax \le 0$ pro nějaké $x \neq o$. Pak $0 \ge x^T Ax = x^T U^T Ux = (Ux)^T Ux = \|Ux\|_2^2$. Tedy musí $Ux = o$, ale sloupce $U$ jsou lineárně nezávislé, a tak $x = o$, spor.

Pro positivní semidefinitnost máme následující charakterizaci (důkaz je analogický):

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Věta 11.8 — Charakterizace positivní semidefinitnosti)</span></p>

Buď $A \in \mathbb{R}^{n \times n}$ symetrická. Pak následující podmínky jsou ekvivalentní:

1. $A$ je positivně semidefinitní,
2. vlastní čísla $A$ jsou nezáporná,
3. existuje matice $U \in \mathbb{R}^{m \times n}$ taková, že $A = U^T U$.

</div>

### Metody na testování positivní definitnosti

Nyní se zaměříme na konkrétní metody pro testování positivní definitnosti. Řada z nich vychází z následujícího rekurentního vztahu.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Věta 11.9 — Rekurentní vzoreček na testování positivní definitnosti)</span></p>

Symetrická matice $A = \begin{pmatrix} \alpha & a^T \\ a & \tilde{A} \end{pmatrix}$, kde $\alpha \in \mathbb{R}$, $a \in \mathbb{R}^{n-1}$, $\tilde{A} \in \mathbb{R}^{(n-1) \times (n-1)}$ je positivně definitní právě tehdy, když $\alpha > 0$ a $\tilde{A} - \frac{1}{\alpha}aa^T$ je positivně definitní.

</div>

*Důkaz.* Implikace „$\Rightarrow$": Buď $A$ positivně definitní. Pak $x^T Ax > 0$ pro všechna $x \neq o$, tedy speciálně pro $x = e_1$ dostáváme $\alpha = e_1^T A e_1 > 0$. Dále, buď $\tilde{x} \in \mathbb{R}^{n-1}$, $\tilde{x} \neq o$. Pak

$$\tilde{x}^T \left(\tilde{A} - \tfrac{1}{\alpha}aa^T\right)\tilde{x} = \tilde{x}^T \tilde{A}\tilde{x} - \tfrac{1}{\alpha}(a^T\tilde{x})^2 = \begin{pmatrix} -\tfrac{1}{\alpha}a^T\tilde{x} & \tilde{x}^T \end{pmatrix} \begin{pmatrix} \alpha & a^T \\ a & \tilde{A} \end{pmatrix} \begin{pmatrix} -\tfrac{1}{\alpha}a^T\tilde{x} \\ \tilde{x} \end{pmatrix} > 0.$$

Implikace „$\Leftarrow$": Buď $x = \begin{pmatrix} \beta \\ \tilde{x} \end{pmatrix} \in \mathbb{R}^n$. Pak

$$x^T Ax = \begin{pmatrix} \beta & \tilde{x}^T \end{pmatrix} \begin{pmatrix} \alpha & a^T \\ a & \tilde{A} \end{pmatrix} \begin{pmatrix} \beta \\ \tilde{x} \end{pmatrix} = \alpha\beta^2 + 2\beta a^T\tilde{x} + \tilde{x}^T\tilde{A}\tilde{x} = \tilde{x}^T(\tilde{A} - \tfrac{1}{\alpha}aa^T)\tilde{x} + \left(\sqrt{\alpha}\beta + \tfrac{1}{\sqrt{\alpha}}a^T\tilde{x}\right)^2 \ge 0.$$

Rovnost nastane pouze tehdy, když $\tilde{x} = o$ a druhý čtverec je nulový, tj. $\beta = 0$.

Přestože rekurentní vzoreček je pro testování positivní definitnosti použitelný, větší roli hraje následující Choleského rozklad.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Věta 11.10 — Choleského rozklad)</span></p>

Pro každou positivně definitní matici $A \in \mathbb{R}^{n \times n}$ existuje jediná dolní trojúhelníková matice $L \in \mathbb{R}^{n \times n}$ s kladnou diagonálou taková, že $A = LL^T$.

</div>

*Důkaz.* Matematickou indukcí podle $n$. Pro $n = 1$ máme $A = (a_{11})$ a $L = (\sqrt{a_{11}})$.

Indukční krok $n \leftarrow n - 1$. Mějme $A = \begin{pmatrix} \alpha & a^T \\ a & \tilde{A} \end{pmatrix}$. Podle věty 11.9 je $\alpha > 0$ a $\tilde{A} - \frac{1}{\alpha}aa^T$ je positivně definitní. Tedy dle indukčního předpokladu existuje dolní trojúhelníková matice $\tilde{L} \in \mathbb{R}^{(n-1) \times (n-1)}$ s kladnou diagonálou taková, že $\tilde{A} - \frac{1}{\alpha}aa^T = \tilde{L}\tilde{L}^T$. Potom $L = \begin{pmatrix} \sqrt{\alpha} & o^T \\ \frac{1}{\sqrt{\alpha}}a & \tilde{L} \end{pmatrix}$, neboť

$$LL^T = \begin{pmatrix} \sqrt{\alpha} & o^T \\ \frac{1}{\sqrt{\alpha}}a & \tilde{L} \end{pmatrix} \begin{pmatrix} \sqrt{\alpha} & \frac{1}{\sqrt{\alpha}}a^T \\ o & \tilde{L}^T \end{pmatrix} = \begin{pmatrix} \alpha & a^T \\ a & \frac{1}{\alpha}aa^T + \tilde{L}\tilde{L}^T \end{pmatrix} = A.$$

Choleského rozklad existuje i pro positivně semidefinitní matice, ale není jednoznačný.

**Algoritmus Choleského rozkladu.** Věta 11.10 byla spíše existenčního charakteru, nicméně sestrojit Choleského rozklad je velkou jednoduché. Základní idea je vyjít z rovnice $A = LL^T$ a postupně porovnávat shora prvky v prvním sloupci matice nalevo a napravo, pak ve druhém sloupci atd.

Předpokládejme, že máme spočítaný prvních $k - 1$ sloupcí matice $L$. Ze vztahu $A = LL^T$ odvodíme pro prvek na pozici $(k, k)$:

$$a_{kk} = \sum_{j=1}^{k} \ell_{kj}^2, \quad \text{tedy} \quad \ell_{kk} = \sqrt{a_{kk} - \sum_{j=1}^{k-1} \ell_{kj}^2}.$$

A pro prvek na pozici $(i, k)$, kde $i > k$:

$$a_{ik} = \sum_{j=1}^{k} \ell_{ij}\ell_{kj}, \quad \text{tedy} \quad \ell_{ik} = \frac{1}{\ell_{kk}} \left(a_{ik} - \sum_{j=1}^{k-1} \ell_{ij}\ell_{kj}\right).$$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Algoritmus 11.11 — Choleského rozklad)</span></p>

Vstup: symetrická matice $A \in \mathbb{R}^{n \times n}$.

1. $L \coloneqq 0_n$,
2. **for** $k \coloneqq 1$ **to** $n$ **do** &emsp;&emsp; // v $k$-tém cyklu určíme hodnoty $L_{*k}$
3. &emsp; **if** $a_{kk} - \sum_{j=1}^{k-1} \ell_{kj}^2 \le 0$ **then return** „$A$ není positivně definitní",
4. &emsp; $\ell_{kk} \coloneqq \sqrt{a_{kk} - \sum_{j=1}^{k-1} \ell_{kj}^2}$,
5. &emsp; **for** $i \coloneqq k + 1$ **to** $n$ **do**
6. &emsp;&emsp; $\ell_{ik} \coloneqq \frac{1}{\ell_{kk}} \left(a_{ik} - \sum_{j=1}^{k-1} \ell_{ij}\ell_{kj}\right)$,
7. &emsp; **end for**
8. **end for**

Výstup: matice $L$ splňující $A = LL^T$ nebo informace, že $A$ není positivně definitní.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 11.12)</span></p>

Choleského rozklad matice $A$:

$$\begin{pmatrix} 2 & 0 & 0 \\ -1 & 3 & 0 \\ 2 & 1 & 1 \end{pmatrix} \begin{pmatrix} 2 & -1 & 2 \\ 0 & 3 & 1 \\ 0 & 0 & 1 \end{pmatrix} = \begin{pmatrix} 4 & -2 & 4 \\ -2 & 10 & 1 \\ 4 & 1 & 6 \end{pmatrix}.$$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 11.13 — Použití Choleského rozkladu pro řešení soustav)</span></p>

Použití Choleského rozkladu pro řešení soustavy $Ax = b$ s positivně definitní maticí $A$. Pokud máme rozklad $A = LL^T$, pak soustava má tvar $L(L^T x) = b$. Nejprve vyřešíme soustavu $Ly = b$ pomocí dopředné substituce, potom $L^T x = y$ pomocí zpětné substituce. Postup je tedy následující:

1. Najdi Choleského rozklad $A = LL^T$.
2. Najdi řešení $y^*$ soustavy $Ly = b$ pomocí dopředné substituce.
3. Najdi řešení $x^*$ soustavy $L^T x = y^*$ pomocí zpětné substituce.

Tento postup je řádově o $50\%$ rychlejší než řešení Gaussovou eliminací.

Choleského rozklad můžeme použít i k invertování positivně definitních matic, protože $A^{-1} = (LL^T)^{-1} = (L^{-1})^T L^{-1}$ a inverze k dolní trojúhelníkové matici $L$ se najde snadno.

</div>

Rekurentní vzoreček má ještě další důsledky, které vyjadřují, jak testovat positivní definitnost pomocí Gaussovy–Jordanovy eliminace a pomocí determinantů.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Tvrzení 11.14 — Gaussova eliminace a positivní definitnost)</span></p>

Symetrická matice $A \in \mathbb{R}^{n \times n}$ je positivně definitní právě tehdy, kdy ji Gaussova eliminace převede do odstupňovaného tvaru s kladnou diagonálou za použití pouze elementární úpravy přičtení násobku řádku s pivotem k jinému řádku pod ním.

</div>

*Důkaz.* Mějme $A = \begin{pmatrix} \alpha & a^T \\ a & \tilde{A} \end{pmatrix}$ positivně definitní. První krok Gaussovy eliminace převede matici na tvar $\begin{pmatrix} \alpha & a^T \\ o & \tilde{A} - \frac{1}{\alpha}aa^T \end{pmatrix}$, stačí od druhého blokového řádku odečíst $\frac{1}{\alpha}a$-násobek prvního řádku. Podle věty 11.9 je $\alpha > 0$ a $\tilde{A} - \frac{1}{\alpha}aa^T$ je zase positivně definitní, takže můžeme pokračovat induktivně dál.

Nyní naopak předpokládejme, že Gaussova eliminace převede matici $A$ do požadovaného tvaru. V prvním kroku ji opět upraví na tvar $\begin{pmatrix} \alpha & a^T \\ o & \tilde{A} - \frac{1}{\alpha}aa^T \end{pmatrix}$, kde $\alpha > 0$. Matematickou indukcí podle velikosti matice můžeme předpokládat, že matice $\tilde{A} - \frac{1}{\alpha}aa^T$ je positivně definitní. Tudíž i matice $A$ je positivně definitní podle věty 11.9.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Tvrzení 11.15 — Sylvestrovo kriterium positivní definitnosti)</span></p>

Symetrická matice $A \in \mathbb{R}^{n \times n}$ je positivně definitní právě tehdy, když determinanty všech hlavních vedoucích podmatic $A_1, \ldots, A_n$ jsou kladné, přičemž $A_i$ je levá horní podmatice $A$ velikosti $i$ (tj. vznikne z $A$ odstraněním posledních $n - i$ řádků a sloupců).

</div>

*Důkaz.* Implikace „$\Rightarrow$": Buď $A$ positivně definitní. Pak pro každé $i = 1, \ldots, n$ je $A_i$ positivně definitní, neboť pokud $x^T A_i x \le 0$ pro jisté $x \neq o$, tak $(x^T \; o^T) A \binom{x}{o} = x^T A_i x \le 0$. Tedy $A_i$ má kladná vlastní čísla a její determinant je také kladný (je roven součinu vlastních čísel).

Implikace „$\Leftarrow$": Během Gaussovy eliminace matice $A$ jsou všechny pivoty kladné, neboť pokud je $i$-tý pivot nekladný, pak $\det(A_i) \le 0$. Podle tvrzení 11.14 je tedy $A$ positivně definitní.

Z nezápornosti determinantů všech hlavních vedoucích podmatic ještě pozitivní semidefinitnost nevyplývá (najděte takový příklad!). Analogie Sylvestrovy podmínky pro positivně semidefinitní matice je následující:

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Tvrzení 11.16 — Sylvestrovo kriterium positivní semidefinitnosti)</span></p>

Symetrická matice $A \in \mathbb{R}^{n \times n}$ je positivně semidefinitní právě tehdy, když determinanty všech hlavních podmatic jsou nezáporné, přičemž hlavní podmatice je matice, která vznikne z $A$ odstraněním určitého počtu (i nulového) řádků a sloupců s týmiž indexy.

</div>

*Důkaz.* Je-li $A$ positivně semidefinitní, pak zřejmě hlavní podmatice jsou také positivně semidefinitní, a tudíž mají nezáporný determinant ($=$ součin vlastních čísel).

Důkaz opačné implikace provedeme matematickou indukcí. Pro $n = 1$ je tvrzení zřejmé. Indukční krok $n \leftarrow n - 1$. Pro spor buď $\lambda < 0$ vlastní číslo $A$, a nechť $x$ je odpovídající vlastní vektor znormovaný tak, že $\|x\|_2 = 1$. Jsou-li všechna ostatní vlastní čísla kladná, pak $\det(A) < 0$, a jsme hotovi. V opačném případě buď $\mu \le 0$ další vlastní číslo $A$ a buď $y$, $\|y\|_2 = 1$, odpovídající vlastní vektor. Protože $x \perp y$, nyní nalezneme $\alpha \in \mathbb{R}$ takové, že vektor $z \coloneqq x + \alpha y$ má aspoň jednu složku nulovou; nechť je to $i$-tá. Pak $z^T Az = (x + \alpha y)^T A(x + \alpha y) = \lambda x^T x + \alpha^2 \mu y^T y = \lambda + \alpha^2 \mu < 0$. Nechť $A'$ vznikne $A$ odstraněním $i$-tého řádku a sloupce, a $z'$ nechť vznikne z vektoru $z$ odstraněním $i$-té složky. Pak $z'^T A' z' = z^T Az < 0$, tudíž hlavní podmatice $A'$ není positivně semidefinitní a aplikujeme indukční předpoklad.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Poznámka 11.17 — Výpočetní složitost)</span></p>

Porovnáme výpočetní složitost jednotlivých metod na testování positivní definitnosti. Podle poznámky 2.19 je asymptotická složitost Gaussovy eliminace $\frac{2}{3}n^3$. Výpočet determinantu má stejnou složitost, proto Sylvestrovo kriterium vyžaduje řádově

$$\sum_{k=1}^{n} \frac{2}{3}k^3 = \frac{2}{3} \cdot \frac{1}{4} n^2(n+1)^2$$

operací, což je asymptoticky $\frac{1}{6}n^4$. Sylvestrovo kriterium se tedy pro praktické použití nehodí. Rekurentní vzoreček stojí řádově

$$\sum_{k=1}^{n} 2k^2 = \frac{2}{6}n(n+1)(2n+1)$$

operací, což dává stejnou složitost jako pro Gaussovu eliminaci, tj. $\frac{2}{3}n^3$. Konečně Choleského rozklad potřebuje

$$\sum_{k=1}^{n} 2k + (n-k)2k = n(n+1) + n^2(n+1) - \frac{2}{6}n(n+1)(2n+1)$$

operací. Asymptoticky tedy stojí pouze $\frac{1}{3}n^3$ operací a je proto výpočetně nejlepší metodou.

</div>

Přestože jsme uvedli několik metod na testování positivní definitnosti, některé jsou si dost podobné. Důkaz tvrzení 11.14 ukazuje, že rekurentní vzoreček a Gaussova eliminace fungují v podstatě stejně. A pokud počítáme determinanty Gaussovou eliminací, tak i Sylvestrovo pravidlo je varianta prvních dvou. Naproti tomu Choleského rozklad je metoda principiálně odlišná.

### Aplikace

Nejprve ukážeme, že pomocí positivně definitních matic můžeme popsat všechny možné skalární součiny na prostoru $\mathbb{R}^n$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Věta 11.18 — Skalární součin a positivní definitnost)</span></p>

Operace $\langle x, y \rangle$ je skalárním součinem v $\mathbb{R}^n$ právě tehdy, když má tvar $\langle x, y \rangle = x^T Ay$ pro nějakou positivně definitní matici $A \in \mathbb{R}^{n \times n}$.

</div>

*Důkaz.* Implikace „$\Rightarrow$": Definujme matici $A \in \mathbb{R}^{n \times n}$ předpisem $a_{ij} = \langle e_i, e_j \rangle$, kde $e_i, e_j$ jsou standardní jednotkové vektory. Matice $A$ je zjevně symetrická a je jednotková. Nyní podle linearity skalárního součinu v první i druhé složce můžeme psát

$$\langle x, y \rangle = \left\langle \sum_{i=1}^n x_i e_i, \sum_{j=1}^n y_j e_j \right\rangle = \sum_{i=1}^n \sum_{j=1}^n x_i y_j \langle e_i, e_j \rangle = \sum_{i=1}^n \sum_{j=1}^n x_i y_j a_{ij} = x^T Ay.$$

Matice $A$ musí být positivně definitní, neboť z definice skalárního součinu $x^T Ax = \langle x, x \rangle \ge 0$ a nulové jen pro $x = o$.

Implikace „$\Leftarrow$": Nechť $A$ je positivně definitní. Pak $\langle x, y \rangle = x^T Ay$ tvoří skalární součin: $\langle x, x \rangle = x^T Ax \ge 0$ a nulové jen pro $x = o$, je lineární v první složce a je symetrický neboť $\langle x, y \rangle = x^T Ay = (x^T Ay)^T = y^T A^T x = y^T Ax = \langle y, x \rangle$.

Víme, že skalární součin indukuje normu (definice 8.5). Norma indukovaná výše zmíněným skalárním součinem je $\|x\| = \sqrt{x^T Ax}$. V této normě je jednotková koule elipsoid (viz příklad 12.22). Pro $A = I_n$ dostáváme standardní skalární součin v $\mathbb{R}^n$ a eukleidovskou normu.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Poznámka 11.19)</span></p>

Přestože nestandardní skalární součin $\langle x, y \rangle = x^T Ay$ může vypadat podivně, jeho vztah ke standardnímu je velmi blízký. Protože matice $A$ je positivně definitní, lze rozložit jako $A = R^T R$, kde $R$ je regulární. Buď $B$ báze tvořená sloupci matice $R^{-1}$, tudíž $R = {}_B[id]_{\text{kan}}$ je matice přechodu od $B$ do kanonické báze. Nyní $x^T Ay = x^T R^T R y = (Rx)^T(Ry) = [x]_B^T [y]_B$. To ukazuje, že nestandardní skalární součin lze vyjádřit jako standardní skalární součin vzhledem k určité bázi.

</div>

Další aplikací je odmocnina z matice. Pro positivně semidefinitní matice můžeme zavést positivně semidefinitní odmocninu $\sqrt{A}$. Odmocnina je dokonce jednoznačná.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Tvrzení 11.20 — Odmocnina z matice)</span></p>

Pro každou positivně semidefinitní matici $A \in \mathbb{R}^{n \times n}$ existuje positivně semidefinitní matice $B \in \mathbb{R}^{n \times n}$ taková, že $B^2 = A$.

</div>

*Důkaz.* Nechť $A$ má spektrální rozklad $A = Q\Lambda Q^T$, kde $\Lambda = \operatorname{diag}(\lambda_1, \ldots, \lambda_n)$, $\lambda_1, \ldots, \lambda_n \ge 0$. Definujme diagonální matici $\Lambda' = \operatorname{diag}(\sqrt{\lambda_1}, \ldots, \sqrt{\lambda_n})$ a matici $B = Q\Lambda' Q^T$. Pak $B^2 = Q\Lambda' Q^T Q\Lambda' Q^T = Q\Lambda'^2 Q^T = Q\Lambda Q^T = A$.

Zde je namístě porovnat odmocninu z matice s maticovými funkcemi z příkladu 10.47. Odmocninu lze vyjádřit nekonečným rozvojem pouze v malém okolí daného kladného čísla, nicméně tam, kde existuje, se budou obě definice shodovat.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Poznámka 11.21 — Positivní definitnost a optimalizace)</span></p>

Positivní (semi-)definitnost se vyskytuje v optimalizaci při určování minima funkce $f \colon \mathbb{R}^n \to \mathbb{R}$. Matice, která zde vystupuje, je tzv. hessián, matice druhých parciálních derivací. Za předpokladu, že jsme v bodě $x^* \in \mathbb{R}^n$ s nulovým gradientem, pak positivní definitnost dává postačující podmínku pro to, aby $x^*$ bylo lokální minimum, a naopak positivní semidefinitnost dává nutnou podmínku. Jedná se o zobecnění jednorozměrného případu, kdy reálná hladká funkce $f \colon \mathbb{R} \to \mathbb{R}$ má v bodě $x^* \in \mathbb{R}$ lokální minimum, pokud její derivace v bodě $a$ je nulová a druhá derivace kladná.

Hessián se podobně používá i při určování konvexity funkce. Positivní definitnost na nějaké otevřené konvexní množině implikuje konvexitu funkce $f$.

Positivně definitní matice hrají v optimalizaci ještě jednu důležitou roli. Semidefinitní program je taková optimalizační úloha, při níž hledáme minimum lineární funkce za podmínky na positivní semidefinitnost matice, jejíž prvky jsou lineární funkcí proměnných. Formálně, jedná se o úlohu

$$\min \; c^T x \quad \text{za podmínky } A_0 + \sum_{i=1}^m A_i x_i \text{ je positivně semidefinitní},$$

kde $c \in \mathbb{R}^m$, $A_0, A_1, \ldots, A_m \in \mathbb{R}^{n \times n}$ jsou dány a $x = (x_1, \ldots, x_m)^T$ je vektor proměnných. Semidefinitními programy dokážeme nejen modelovat větší třídu úloh než lineárními programy, ale stále jsou řešitelné efektivně za rozumný čas. Umožnily mj. velký pokrok na poli kombinatorické optimalizace, protože řada výpočetně složitých problémů se dá rychle a těsně aproximovat právě pomocí vhodných semidefinitních programů.

</div>

Výskyt positivně (semi-)definitních matic je ještě širší. Například ve statistice se setkáváme s tzv. kovarianční a korelační maticí. Obě dávají jistou informaci o závislosti mezi $n$ náhodnými veličinami a, ne náhodou, jsou vždy positivně semidefinitní.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 11.22 — Určování struktury bílkovin)</span></p>

Jedna ze základních úloh modelování bílkovin je určení trojrozměrné struktury bílkovin. Typický postup je určení matice vzdáleností jednotlivých atomů pomocí jaderné magnetické rezonance a pak odvození struktury.

Buď $X \in \mathbb{R}^{n \times 3}$ matice pozic jednotlivých atomů, to jest, řádek $X_{i*}$ udává souřadnice $i$-tého atomu v prostoru. Matici $D \in \mathbb{R}^{n \times n}$ si označíme vzdálenosti jednotlivých atomů, tedy $d_{ij} = $ vzdálenost $i$-tého a $j$-tého atomu. Jestliže známe $X$, tak $D$ spočítáme. Posuneme souřadný systém, aby $n$-tý atom byl v počátku, tj. $X_{n*} = (0, 0, 0)$ a z matice $X$ odstraníme poslední řádek, který je nyní redundantní. Označme pomocnou matici $D^* \coloneqq XX^T$ a souřadnice libovolných dvou atomů $u \coloneqq X_{i*}$, $v \coloneqq X_{j*}$. Vztah matic $D$ a $D^*$ je tento:

$$d_{ij}^2 = \|u - v\|^2 = \langle u - v, u - v \rangle = \langle u, u \rangle + \langle v, v \rangle - 2\langle u, v \rangle = d_{ii}^* + d_{jj}^* - 2d_{ij}^*.$$

Tímto předpisem z matice $D$ spočítáme matici $D^*$:

$$d_{ij}^* = \frac{1}{2}(d_{ii}^* + d_{jj}^* - d_{ij}^2) = \frac{1}{2}(d_{in}^2 + d_{jn}^2 - d_{ij}^2).$$

Protože $D^*$ je symetrická a positivně semidefinitní, ze spektrálního rozkladu $D^* = Q\Lambda Q^T$, kde $\Lambda = \operatorname{diag}(\lambda_1, \lambda_2, \lambda_3)$, sestrojíme hledanou matici $X = Q \cdot \operatorname{diag}(\sqrt{\lambda_1}, \sqrt{\lambda_2}, \sqrt{\lambda_3})$. Pak totiž máme $D^* = XX^T$.

Jiný problém z tohoto oboru je tzv. *Prokrustův problém*, v němž porovnáváme dvě struktury bílkovin jak moc jsou podobné. Označme matice dvou struktur $X, Y$. Lineární zobrazení $f$ ortogonální maticí $Q$ zachovává úhly a vzdálenosti, tudíž $YQ$ odpovídá té samé struktuře jako $Y$, pouze nějakým způsobem otočené či zrcadlené. Chceme-li určit podobnost obou struktur, hledáme takovou ortogonální matici $Q \in \mathbb{R}^{3 \times 3}$, aby matice $X$ a $YQ$ byly „co nejbližší". Matematická formulace vede na optimalizační úlohu minimalizovat hodnotu maticové normy $\|X - YQ\|$ na množině ortogonálních matic $Q$.

</div>

### Shrnutí ke kapitole 11

Positivně definitní matice je speciální typ matice, která se nicméně vyskytuje v rozličných situacích:

- každý skalární součin v prostoru $\mathbb{R}^n$ je tvaru $\langle x, y \rangle = x^T Ay$ pro nějakou positivně definitní matici $A$,
- funkce $f \colon \mathbb{R}^n \to \mathbb{R}$ je konvexní, pokud její hessián je positivně definitní,
- a další.

Positivně definitní matici definujeme jako takovou symetrickou matici $A$, pro kterou funkce $f(x) = x^T Ax$ je nezáporná a nulová pouze v počátku. Alternativně ji můžeme charakterizovat jako matici, která má kladná vlastní čísla. Ještě jinak lze positivně definitní matice nahlédnout tak, že mají rozklad $A = U^T U$, kde $U$ je regulární. Dokonce lze po matici $U$ požadovat, aby byla horní trojúhelníková s kladnou diagonálou (pak je i jednoznačná). To dává vzniku i efektivnímu způsobu na testování positivní definitnosti, tzv. Choleského rozkladu (bývá zvykem ho psát ve tvaru $A = LL^T$, kde $L$ je dolní trojúhelníková matice). Navíc rozklad $A = U^T U$ najde uplatnění při řešení soustav lineárních rovnic či jiných výpočtech s maticí $A$.

## Kapitola 12 — Kvadratické formy

Slovo *lineární* ve výrazu „lineární algebra" neznamená, že se obor zabývá jen lineárními objekty jako třeba přímkami a rovinami při aplikaci v geometrii. V této kapitole si podrobněji všimneme kvadratických forem. V zásadě jsme se s nimi setkali již u eukleidovské normy (str. 130) a positivní definitnosti (definice 11.1). V této kapitole probereme kvadratické formy podrobněji. Ukážeme souvislosti s positivní (semi-)definitností, znaménky vlastních čísel a popisem určitých geometrických objektů.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 12.1 — Motivace ke kvadratickým formám)</span></p>

Pro začátek si můžeme kvadratickou formu představit jako polynom $n$ proměnných, kde součet stupňů každého členu je přesně dva. Čili

$$f(x) = 5x^2$$

je kvadratická forma s jednou proměnnou a

$$f(x_1, x_2) = 5x_1^2 - 3x_1 x_2 + 12x_2^2$$

je kvadratická forma se dvěma proměnnými. Obecný předpis pro takovýto polynom s $n$ proměnnými $x = (x_1, \ldots, x_n)^T$ je

$$f(x) = \sum_{i=1}^{n} \sum_{j=1}^{n} a_{ij} x_i x_j = x^T A x,$$

kde $A$ je matice $n \times n$ příslušných koeficientů. Kompaktní maticový zápis $x^T Ax$ budeme často používat.

</div>

### 12.1 Bilineární a kvadratické formy

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Definice 12.2 — Bilineární a kvadratická forma)</span></p>

Buď $V$ vektorový prostor nad $\mathbb{T}$. *Bilineární forma* je zobrazení $b \colon V^2 \to \mathbb{T}$, které je lineární v první i druhé složce, tj.

$$b(\alpha u + \beta v, w) = \alpha b(u, w) + \beta b(v, w), \quad \forall \alpha, \beta \in \mathbb{T}, \; \forall u, v, w \in V,$$

$$b(w, \alpha u + \beta v) = \alpha b(w, u) + \beta b(w, v), \quad \forall \alpha, \beta \in \mathbb{T}, \; \forall u, v, w \in V.$$

Bilineární forma se nazývá *symetrická*, pokud $b(u, v) = b(v, u)$ pro všechna $u, v \in V$. Zobrazení $f \colon V \to \mathbb{T}$ je *kvadratická forma*, pokud se dá vyjádřit $f(u) = b(u, u)$ pro nějakou symetrickou bilineární formu $b$.

Je lehké nahlédnout, že vždy platí $b(o, v) = b(v, o) = 0$, $f(o) = 0$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 12.3)</span></p>

- Každý reálný skalární součin je bilineární formou, tedy pro libovolnou matici $A \in \mathbb{R}^{n \times n}$ je zobrazení $b(x, y) = x^T Ay$ bilineární formou. Pro symetrickou matici $A$ je pak kvadratickou formou zobrazení $f(x) = x^T Ax$.

  Speciálně, v prostoru $\mathbb{R}^1$ je bilineární formou jakékoli zobrazení $b \colon \mathbb{R}^2 \to \mathbb{R}$ s předpisem $b(x, y) = axy$, kde $a \in \mathbb{R}$ je konstanta. Příslušná kvadratická forma je pak kvadratická funkce jedné proměnné $f(x) = ax^2$. Kvadratické formy na $\mathbb{R}^n$ lze tedy chápat jako zobecnění kvadratické funkce z jedné na $n$ proměnných.

- Komplexní skalární součin není bilineární formou, protože není lineární v druhé složce. Formy takovéhoto typu se nazývají sesquilineární.

- Buď $V = \mathbb{R}^2$. Pak $b(x, y) = x_1 y_1 + 2x_1 y_2 + 4x_2 y_1 + 10x_2 y_2$ je příkladem bilineární formy, $b'(x, y) = x_1 y_1 + 3x_1 y_2 + 3x_2 y_1 + 10x_2 y_2$ je příkladem symetrické bilineární formy a $f(x) = b'(x, x) = x_1^2 + 6x_1 x_2 + 10x_2^2$ odpovídající kvadratické formy.

</div>

V této kapitole se budeme převážně zabývat kvadratickými formami, i když bilineární formy jsou také zajímavé, například právě vztahem se skalárním součinem.

V analogii s teorií lineárních zobrazení, i bilineární formy jsou jednoznačně určeny obrazy bází a dají se vyjádřit maticově. Čtenáři proto doporučujeme porovnat pojmy a výsledky této sekce se sekcí 6.2 a uvědomit si jistou paralelu.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Poznámka 12.5 — Motivace k maticím forem)</span></p>

Buď $b \colon V^2 \to \mathbb{T}$ bilineární forma a $B = \lbrace w_1, \ldots, w_n \rbrace$ báze prostoru $V$. Mějme libovolné dva vektory $u, v \in V$ a nechť mají v bázi $B$ vyjádření $u = \sum_{i=1}^n x_i w_i$, $v = \sum_{i=1}^n y_i w_i$. Z definice bilineární formy pak obraz vektorů je

$$b(u, v) = b\!\left(\sum_{i=1}^n x_i w_i, \sum_{j=1}^n y_j w_j\right) = \sum_{i=1}^n \sum_{j=1}^n x_i y_j b(w_i, w_j).$$

Vidíme, že celá bilineární forma je vlastně určena tím, kam se zobrazí všechny dvojice bázických vektorů. To nás navíc motivuje umístit tyto hodnoty $b(w_i, w_j)$ do matice a pracovat s maticovou reprezentací.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Definice 12.6 — Matice bilineární a kvadratické formy)</span></p>

Buď $b \colon V^2 \to \mathbb{T}$ bilineární forma a $B = \lbrace w_1, \ldots, w_n \rbrace$ báze prostoru $V$. Pak definujeme matici $A \in \mathbb{T}^{n \times n}$ bilineární formy vzhledem k bázi $B$ předpisem $a_{ij} = b(w_i, w_j)$. Matice kvadratické formy $f \colon V \to \mathbb{T}$ je definována jako matice libovolné symetrické bilineární formy indukující $f$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Věta 12.7 — Maticové vyjádření forem)</span></p>

Buď $B$ báze vektorového prostoru $V$ a buď $b$ bilineární forma na $V$. Pak $A$ je matice formy $b$ vzhledem k bázi $B$ právě tehdy, když pro každé $u, v \in V$ platí

$$b(u, v) = [u]_B^T A [v]_B.$$

Dále, je-li $b$ symetrická forma, pak odpovídající kvadratická forma $f$ pro každé $u \in V$ splňuje

$$f(u) = [u]_B^T A [u]_B.$$

</div>

*Důkaz.* Označme $x \coloneqq [u]_B$, $y \coloneqq [v]_B$, a nechť $B$ se skládá z vektorů $w_1, \ldots, w_n$. Je-li $A$ matice formy $b$, tak

$$b(u, v) = b\!\left(\sum_{i=1}^n x_i w_i, \sum_{j=1}^n y_j w_j\right) = \sum_{i=1}^n \sum_{j=1}^n x_i y_j b(w_i, w_j) = \sum_{i=1}^n \sum_{j=1}^n x_i y_j a_{ij} = x^T Ay.$$

Naopak, pokud platí (12.1) pro každé $u, v \in V$, tak dosazením $u \coloneqq w_i$, $v \coloneqq w_j$ dostaneme $b(w_i, w_j) = [w_i]_B^T A [w_j]_B = e_i^T A e_j = a_{ij}$ pro všechna $i, j = 1, \ldots, n$.
Konečně, $f(u) = b(u, u) = x^T Ax$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(Důsledek 12.8)</span></p>

Buď $B = \lbrace w_1, \ldots, w_n \rbrace$ báze vektorového prostoru $V$ nad $\mathbb{T}$ a buď $A \in \mathbb{T}^{n \times n}$. Pak existuje jediná bilineární forma $b \colon V^2 \to \mathbb{T}$ taková, že $b(w_i, w_j) = a_{ij}$ pro všechna $i, j = 1, \ldots, n$.

</div>

*Důkaz.* „Existence." Stačí ověřit, že zobrazení $b \colon V^2 \to \mathbb{T}$ dané předpisem $b(u, v) = [u]_B^T A [v]_B$ splňuje podmínky bilineární formy. To se nahlédne snadno, neboť zobrazení $u \mapsto [u]_B$ je lineární (srov. tvrzení 6.38). „Jednoznačnost." Z (12.1) plyne, že pro každé $u, v \in V$ je $b(u, v) = [u]_B^T A [v]_B$, tedy obrazy jsou jednoznačně dány.

Buď $B$ pevná báze prostoru $V$ dimenze $n$. Každé bilineární formě tedy odpovídá jednoznačně matice $A \in \mathbb{T}^{n \times n}$, a naopak každé matici $A \in \mathbb{T}^{n \times n}$ odpovídá jednoznačně bilineární forma. Existuje tudíž vzájemně jednoznačný vztah mezi množinou bilineárních forem a prostorem matic $\mathbb{T}^{n \times n}$. Jedná se navíc o isomorfismus, protože bilineární formy tvoří vektorový prostor s přirozeně definovaným součtem a násobky (srov. prostor $\mathcal{F}$ ze str. 79).

Ve vektorovém prostoru $\mathbb{T}^n$ mají bilineární formy speciální tvar.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(Důsledek 12.9)</span></p>

Nechť charakteristika tělesa $\mathbb{T}$ není 2. Pak každá bilineární forma na $\mathbb{T}^n$ se dá vyjádřit ve tvaru

$$b(x, y) = x^T Ay$$

pro určitou matici $A \in \mathbb{T}^{n \times n}$, a každá kvadratická forma na $\mathbb{T}^n$ se dá vyjádřit ve tvaru

$$f(x) = x^T Ax$$

pro určitou symetrickou matici $A \in \mathbb{T}^{n \times n}$.

</div>

*Důkaz.* Stačí vzít $A$ jako matici formy vzhledem ke kanonické bázi. Pak $b(x, y) = [x]_{\text{kan}}^T A [y]_{\text{kan}} = x^T Ay$. Pro kvadratickou formu pak platí $f(x) = b(x, x) = x^T Ax$. Pokud $A$ není symetrická, nahradíme ji symetrickou maticí $\frac{1}{2}(A + A^T)$ ve smyslu poznámky 11.2, protože $x^T Ax = x^T \frac{1}{2}(A + A^T)x$. Používáme zde úzus $2 \equiv 1 + 1$. Protože charakteristika tělesa není 2, platí $1 + 1 \neq 0$ a matici můžeme zkonstruovat.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 12.10)</span></p>

Uvažme bilineární formu na $\mathbb{R}^2$

$$b(x, y) = x_1 y_1 + 2x_1 y_2 + 4x_2 y_1 + 10x_2 y_2.$$

Matice $b$ vzhledem ke kanonické bázi je $A = \begin{pmatrix} 1 & 2 \\ 4 & 10 \end{pmatrix}$, což snadno nahlédneme i z vyjádření

$$b(x, y) = x^T Ay = \begin{pmatrix} x_1 & x_2 \end{pmatrix} \begin{pmatrix} 1 & 2 \\ 4 & 10 \end{pmatrix} \begin{pmatrix} y_1 \\ y_2 \end{pmatrix}.$$

Tato bilineární forma není symetrická, narozdíl od bilineární formy

$$b'(x, y) = x_1 y_1 + 3x_1 y_2 + 3x_2 y_1 + 10x_2 y_2.$$

Matice $b'$ vzhledem ke kanonické bázi je $A' = \begin{pmatrix} 1 & 3 \\ 3 & 10 \end{pmatrix}$, tedy $b'(x, y) = x^T A' y$. Odpovídající kvadratická forma splňuje

$$f'(x) = b'(x, x) = x^T A' x = \begin{pmatrix} x_1 & x_2 \end{pmatrix} \begin{pmatrix} 1 & 3 \\ 3 & 10 \end{pmatrix} \begin{pmatrix} x_1 \\ x_2 \end{pmatrix}.$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Poznámka 12.11 — Proč symetrická bilineární forma)</span></p>

Mohli bychom si klást otázku, proč kvadratickou formu definujeme pouze pomocí symetrických bilineárních forem. Vždyť nic nám nebrání zavést $f(u) = b(u, u)$ i pro nesymetrickou bilineární formu $b$. Důvod je podobný, jaký jsme uvedli u positivně definitních matic, viz poznámka 11.2. Můžeme zavést bilineární formu $b_s(u, v) \coloneqq \frac{1}{2}\bigl(b(u, v) + b(v, u)\bigr)$, která již bude symetrická. Navíc, jak se snadno nahlédne, obě formy $b$, $b_s$ indukují stejnou kvadratickou formu $f$. Zde ovšem těleso $\mathbb{T}$ nesmí mít charakteristiku 2, jinak by zlomek nedával smysl. Restrikcí na symetrický případ pak také v důsledku máme jednoznačnost matice kvadratických forem, opět za předpokladu, že těleso $\mathbb{T}$ nemá charakteristiku 2.

</div>

Matice forem závisí na volbě báze. Jak se změní matice, když přejdeme k jiné bázi?

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Věta 12.12 — Matice kvadratické formy při změně báze)</span></p>

Buď $A \in \mathbb{T}^{n \times n}$ matice kvadratické formy $f$ vzhledem k bázi $B$ prostoru $V$. Buď $B'$ jiná báze a $S = {}_B[id]_{B'}$ matice přechodu od $B'$ k $B$. Pak matice formy $f$ vzhledem k bázi $B'$ je $S^T AS$ a odpovídá stejné symetrické bilineární formě.

</div>

*Důkaz.* Buď $u, v \in V$ a $b$ symetrická bilineární forma indukující $f$. Pak

$$b(u, v) = [u]_B^T A [v]_B = ({}_B[id]_{B'} \cdot [u]_{B'})^T A ({}_B[id]_{B'} \cdot [v]_{B'}) = [u]_{B'}^T S^T A S [v]_{B'}.$$

Podle věty 12.7 je $S^T AS$ matice formy $b$, a tím i $f$, vzhledem k bázi $B'$.

Různou volbou báze prostoru $V$ dosahujeme různé maticové reprezentace. Naším cílem bude najít takovou bázi, vůči níž je matice co nejjednodušší, tedy diagonální.

Zde je jistá paralela s diagonalizací pro vlastní čísla, kde jsme transformovali matici pomocí podobnosti. Nyní transformujeme matici úpravou $S^T AS$, kde $S$ je regulární. Místo podobnosti nyní máme tzv. *kongruenci*. Jak uvidíme, u kvadratických forem je situace jednodušší — každou matici lze diagonalizovat.

### 12.2 Sylvestrův zákon setrvačnosti

V této sekci nadále uvažujeme reálný prostor $\mathbb{R}^n$ nad $\mathbb{R}$. Z definice je patrné, že matice kvadratické formy je symetrická. Tuto vlastnost budeme velmi potřebovat.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Věta 12.13 — Sylvestrův zákon setrvačnosti)</span></p>

Buď $f(x) = x^T Ax$ kvadratická forma na $\mathbb{R}^n$. Pak existuje báze, vůči níž má $f$ diagonální matici s prvky $1, -1, 0$. Navíc, tato matice je až na pořadí prvků jednoznačná.

</div>

*Důkaz.* „Existence". Protože $A$ je symetrická, tak má spektrální rozklad $A = Q\Lambda Q^T$, kde $\Lambda = \operatorname{diag}(\lambda_1, \ldots, \lambda_n)$. Tedy $\Lambda = Q^T AQ$ je diagonalizace formy. Abychom docílili na diagonále $\pm 1$, tak provedeme ještě úpravu $\Lambda' Q^T A Q \Lambda'$, kde $\Lambda'$ je diagonální matice s prvky $\Lambda'_{ii} = |\lambda_i|^{-1/2}$, pokud $\lambda_i \neq 0$ a $\Lambda'_{ii} = 1$ jinak. Nyní můžeme $Q\Lambda'$ považovat za matici ${}\_{\text{kan}}[id]_B$ přechodu od hledané báze $B$ do kanonické báze. Tudíž bázi $B$ vyčteme ve sloupcích matice $Q\Lambda'$.

„Jednoznačnost". Sporem předpokládejme, že máme dvě různé diagonalizace $D$, $D'$:

$$D = \operatorname{diag}(\underbrace{1, \ldots, 1}_{p}, \underbrace{-1, \ldots, -1}_{q-p}, \underbrace{0, \ldots, 0}_{n-q}), \qquad D' = \operatorname{diag}(\underbrace{1, \ldots, 1}_{s}, \underbrace{-1, \ldots, -1}_{t-s}, \underbrace{0, \ldots, 0}_{n-t}).$$

První nechť odpovídá bázi $B = \lbrace w_1, \ldots, w_n \rbrace$ a druhá bázi $B' = \lbrace w'_1, \ldots, w'_n \rbrace$. Buď $u \in \mathbb{R}^n$ libovolné a nechť má souřadnice $y = [u]_B$, $z = [u]_{B'}$. Pak podle věty 12.7

$$f(u) = [u]_B^T D [u]_B = y^T D y = y_1^2 + \ldots + y_p^2 - y_{p+1}^2 - \ldots - y_q^2 + 0 y_{q+1}^2 + \ldots + 0 y_n^2,$$

$$f(u) = [u]_{B'}^T D' [u]_{B'} = z^T D' z = z_1^2 + \ldots + z_s^2 - z_{s+1}^2 - \ldots - z_t^2 + 0 z_{t+1}^2 + \ldots + 0 z_n^2.$$

Nejprve si povšimněme, že $q = t$. Protože $D = S^T D' S$ pro nějakou regulární $S$, konkrétně pro $S = {}_{B'}[id]_B$, tak matice $D, D'$ mají stejnou hodnost. Tudíž musí $q = t$. Nyní zbývá ukázat, že nutně $p = s$. Bez újmy na obecnosti předpokládejme $p > s$. Definujme prostory $P = \operatorname{span}\lbrace w_1, \ldots, w_p \rbrace$ a $R = \operatorname{span}\lbrace w'_{s+1}, \ldots, w'_n \rbrace$. Pak

$$\dim P \cap R = \dim P + \dim R - \dim(P + R) \ge p + (n - s) - n = p - s \ge 1.$$

Tedy existuje nenulový vektor $u \in P \cap R$ a pro něj máme $u = \sum_{i=1}^p y_i w_i = \sum_{j=s+1}^n z_j w'_j$, z čehož dostáváme

$$f(u) = \begin{cases} y_1^2 + \ldots + y_p^2 > 0, \\ -z_{s+1}^2 - \ldots - z_t^2 \le 0. \end{cases}$$

To je spor.

Báze, vůči níž matice kvadratické formy je diagonální, se nazývá *polární báze*. Tedy báze z věty 12.13 je příkladem polární báze, ale typicky existují i další. Dá se také ukázat, že polární báze existuje nejen pro reálné prostory, ale i pro prostory nad libovolným tělesem charakteristiky různé od 2.

Význam Sylvestrova zákona setrvačnosti nespočívá pouze v existenci diagonalizace, ale zejména v její jednoznačnosti (odtud název „setrvačnost"). Tato jednoznačnost opravňuje k zavedení pojmu *signatura* jako trojice $(p, q, z)$, kde $p$ je počet jedniček, $q$ počet minus jedniček a $z$ počet nul ve výsledné diagonální matici. Navíc má řadu důsledků týkajících se mj. positivní (semi-)definitnosti a vlastních čísel.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(Důsledek 12.15)</span></p>

Buď $A \in \mathbb{R}^{n \times n}$ symetrická a $S^T AS$ převedení na diagonální tvar. Pak počet jedniček resp. minus jedniček resp. nul na diagonále odpovídá počtu kladných resp. záporných resp. nulových vlastních čísel matice $A$.

</div>

*Důkaz.* Stačí uvažovat kvadratickou formu $f(x) = x^T Ax$, která má matici $A$. Z důkazu věty 12.13 (část „existence") je patrné, že jednu diagonalizaci získáme ze spektrálního rozkladu a pro ni tvrzení platí. Díky jednoznačnosti ve tvrzení Sylvestrova zákona setrvačnosti pak musí počty souhlasit i pro jakoukoli jinou diagonalizaci.

Diagonalizací matice $A$ tedy nenajdeme vlastní čísla, ale určíme kolik jich je kladných a kolik záporných.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(Důsledek 12.16)</span></p>

Buď $A \in \mathbb{R}^{n \times n}$ symetrická a $S^T AS$ převedení na diagonální tvar. Pak

1. $A$ je positivně definitní právě tehdy, když $S^T AS$ má kladnou diagonálu,
2. $A$ je positivně semidefinitní právě tehdy, když $S^T AS$ má nezápornou diagonálu.

</div>

*Důkaz.* Z důsledku 12.15 a vztahu mezi positivní (semi-)definitností a vlastními čísly (věta 11.7).

Sylvestrův zákon tedy dává návod, jak jednou metodou rozhodnout o positivní definitnosti resp. positivní semidefinitnosti resp. negativní (semi-)definitnosti v jednom.

#### Diagonalizace matice pomocí elementárních úprav

Zbývá otázka, jak matici kvadratické formy převést na diagonální tvar. Důkaz věty o Sylvestrově zákonu setrvačnosti sice dává návod (přes spektrální rozklad), ale můžeme jednoduše adaptovat elementární maticové úpravy. Co se stane, když symetrickou matici $A$ transformujeme na $EAE^T$, kde $E$ je matice elementární řádkové úpravy? Součinem $EA$ se provede řádková úprava a vynásobením $E^T$ zprava se provede i analogická sloupcová úprava. Základní myšlenka metody na diagonalizaci je tedy aplikovat na matici řádkové úpravy a odpovídající sloupcové úpravy. Tím budeme nulovat prvky pod i nad diagonálou, až matici převedeme na diagonální tvar.

Předpokládejme nejprve, že na pozici pivota je vždy nenulové číslo, a tím pádem se můžeme omezit pouze na druhou elementární úpravu — přičtení $\alpha$-násobku $j$-tého řádku k $i$-tému řádku pod ním. Tato úprava nuluje symetricky prvky pod i napravo od pivota. Navíc nepokazí tu část, která je již upravena. Ve výsledku nutně dostaneme diagonální matici.

Drobná potíž nastane, pokud se během maticových úprav vyskytne na pozici pivota nula. Můžeme ale k prvnímu řádku přičíst druhý řádek a analogicky pro sloupce, což vede na matici s nenulovým pivotem. Tento postup lze aplikovat obecně. Pokud je na pozici pivota nula, přičteme k němu vhodný řádek pod ním a analogicky pro sloupce.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 12.17 — Diagonalizace matice kvadratické formy)</span></p>

Diagonalizujeme matici $A$, na kterou aplikujeme střídavě řádkovou úpravu a potom odpovídající sloupcovou:

$$A = \begin{pmatrix} 1 & 2 & -1 \\ 2 & 5 & -3 \\ -1 & -3 & 2 \end{pmatrix} \sim \begin{pmatrix} 1 & 2 & -1 \\ 0 & 1 & -1 \\ -1 & -3 & 2 \end{pmatrix} \sim \begin{pmatrix} 1 & 0 & -1 \\ 0 & 1 & -1 \\ -1 & -1 & 2 \end{pmatrix} \sim$$

$$\sim \begin{pmatrix} 1 & 0 & -1 \\ 0 & 1 & -1 \\ 0 & -1 & 1 \end{pmatrix} \sim \begin{pmatrix} 1 & 0 & 0 \\ 0 & 1 & -1 \\ 0 & -1 & 1 \end{pmatrix} \sim \begin{pmatrix} 1 & 0 & 0 \\ 0 & 1 & -1 \\ 0 & 0 & 0 \end{pmatrix} \sim \begin{pmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 0 \end{pmatrix}.$$

Vidíme, že matice $A$ má dvě kladná vlastní čísla a jedno nulové, je tedy positivně semidefinitní.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 12.18 — Nalezení polární báze)</span></p>

Pro jednoduchost uvažujme kvadratickou formu $f(x) = x^T Ax$, kde $A \in \mathbb{R}^{n \times n}$ je symetrická. Pokud najdeme matici $S \in \mathbb{R}^{n \times n}$ takovou, že $S^T AS$ je diagonální, pak je polární báze obsažena ve sloupcích matice $S$. Jak ovšem matici $S$ nalézt? Pokud diagonalizujeme matici $A$ pomocí elementárních úprav, tak matice $S$ reprezentuje akumulované sloupcové úpravy. Metoda je nyní nasnadě: upravujeme dvojmatici $(A \mid I_n)$ tak, že na matici $A$ aplikujeme řádkové a sloupcové úpravy, abychom ji diagonalizovali, a na jednotkovou matici aplikujeme pouze sloupcové úpravy. Polární bázi pak vyčteme ve sloupcích matice napravo.

Postup aplikujeme na matici z příkladu 12.17 a jednotlivé úpravy jsou

$$(A \mid I_3) = \begin{pmatrix} 1 & 2 & -1 \mid 1 & 0 & 0 \\ 2 & 5 & -3 \mid 0 & 1 & 0 \\ -1 & -3 & 2 \mid 0 & 0 & 1 \end{pmatrix} \sim \begin{pmatrix} 1 & 0 & -1 \mid 1 & -2 & 0 \\ 0 & 1 & -1 \mid 0 & 1 & 0 \\ -1 & -1 & 2 \mid 0 & 0 & 1 \end{pmatrix} \sim$$

$$\sim \begin{pmatrix} 1 & 0 & 0 \mid 1 & -2 & 1 \\ 0 & 1 & -1 \mid 0 & 1 & 0 \\ 0 & -1 & 1 \mid 0 & 0 & 1 \end{pmatrix} \sim \begin{pmatrix} 1 & 0 & 0 \mid 1 & -2 & -1 \\ 0 & 1 & 0 \mid 0 & 1 & 1 \\ 0 & 0 & 0 \mid 0 & 0 & 1 \end{pmatrix}.$$

Příslušná polární báze se tedy skládá z vektorů $(1, 0, 0)^T$, $(-2, 1, 0)^T$, $(-1, 1, 1)^T$. Pokud z těchto vektorů po sloupcích sestavíme matici $S$, potom $S^T AS$ je diagonální matice s prvky $1, 1, 0$ na diagonále.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 12.19 — Součet čtverců lineárních forem)</span></p>

Uvažujme kvadratickou formu $f(x) = x^T Ax$ se symetrickou maticí $A \in \mathbb{R}^{n \times n}$. Pokud výraz $x^T Ax$ s proměnnými $x_1, \ldots, x_n$ dokážeme vyjádřit jako součet čtverců lineárních forem, potom zjevně $f(x) \ge 0$ pro všechna $x \in \mathbb{R}^n$ a matice $A$ je positivně semidefinitní. Zajímavé je, že platí i opačný směr: Každou kvadratickou formu s positivně semidefinitní maticí lze vyjádřit jako součet čtverců lineárních forem.

Najdeme matici $S$, pro kterou je $S^T AS = D$ diagonální. Pak $A = S^{-T} D S^{-1}$ a substitucí $y \coloneqq S^{-1}x$ dostáváme požadovaný tvar

$$x^T Ax = x^T S^{-T} D S^{-1} x = y^T Dy = \sum_{i=1}^n d_{ii} y_i^2 = \sum_{i=1}^n d_{ii} (S_{i*}^{-1} x)^2.$$

Konkrétně uvažujme matici z příkladu 12.18, kde jsme již nahlédli, že $S^T AS = D$ pro

$$S = \begin{pmatrix} 1 & -2 & -1 \\ 0 & 1 & 1 \\ 0 & 0 & 1 \end{pmatrix}, \quad D = \begin{pmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 0 \end{pmatrix}.$$

Spočítáme

$$S^{-1} = \begin{pmatrix} 1 & 2 & -1 \\ 0 & 1 & -1 \\ 0 & 0 & 1 \end{pmatrix}.$$

Nyní $\sum_{i=1}^n d_{ii}(S_{i*}^{-1}x)^2 = (x_1 + 2x_2 - x_3)^2 + (x_2 - x_3)^2$. Dokázali jsme tedy vyjádřit

$$x^T Ax = x_1^2 + 4x_1 x_2 - 2x_1 x_3 + 5x_2^2 - 6x_2 x_3 + 2x_3^2 = (x_1 + 2x_2 - x_3)^2 + (x_2 - x_3)^2.$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Poznámka 12.20 — Součin positivně definitních matic)</span></p>

Součin positivně definitních matic $A, B \in \mathbb{R}^{n \times n}$ nemusí být positivně definitní matice. Součin $AB$ obecně není symetrickou maticí, a navíc ani symetrizací ve smyslu poznámky 11.2 nemusíme dostat positivně definitní matici.

Zajímavé ale je, že matice $AB$, přestože není nutně symetrická, má stále reálná kladná vlastní čísla. Snadno je to vidět z vyjádření $AB = \sqrt{A} \sqrt{A} B$. Vynásobením $\sqrt{A}^{-1}$ zleva a $\sqrt{A}$ zprava dostaneme podobnou matici $\sqrt{A} B \sqrt{A}$. Ta je symetrická, čili má reálná vlastní čísla, a ze setrvačnosti má stejnou signaturu jako $B$, což znamená kladná vlastní čísla.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Poznámka 12.21 — Blokový test positivní definitnosti)</span></p>

Je úzký vztah mezi diagonalizací matice pomocí elementárních úprav a rekurentním vzorečkem na testování positivní definitnosti (věta 11.9). Pokud uvažujeme matici $A = \begin{pmatrix} \alpha & a^T \\ a & \tilde{A} \end{pmatrix}$ jako matici kvadratické formy a elementárními úpravami vynulujeme prvky pod i napravo od pivota, výsledná blokově diagonální matice se dá maticově vyjádřit jako

$$\begin{pmatrix} \alpha & o^T \\ o & \tilde{A} - \frac{1}{\alpha}aa^T \end{pmatrix}.$$

Tato matice je positivně definitní právě tehdy, když všechny bloky jsou positivně definitní matice, tj. $\alpha > 0$ a $\tilde{A} - \frac{1}{\alpha}aa^T$ je positivně definitní. Tím jsme dostali jiné odvození rekurentního vzorečku.

</div>

### Kuželosečky a kvadriky

Pomocí kvadratických forem lze popisovat geometrické útvary zvané *kvadriky*. To jsou (stručně řečeno) množiny popsané rovnicí $x^T Ax + b^T x + c = 0$, kde $A \in \mathbb{R}^{n \times n}$ je symetrická, $b \in \mathbb{R}^n$, $c \in \mathbb{R}$. Jak vidíme, tato rovnice již není lineární právě díky kvadratickému členu $x^T Ax$. Pomocí různých charakteristik, jako jsou například vlastní čísla resp. signatura matice $A$, můžeme pak snadno klasifikovat jednotlivé geometrické tvary kvadrik. Těmito tvary jsou elipsoidy, paraboloidy, hyperboloidy aj., viz příklady dole.

Speciálním případem kvadrik v prostoru $\mathbb{R}^2$ jsou pak *kuželosečky*. Mezi ně patří elipsy, paraboly či hyperboly.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 12.22 — Elipsoidy)</span></p>

Rovnice $\frac{1}{a^2}x_1^2 + \frac{1}{b^2}x_2^2 = 1$ popisuje v rovině $\mathbb{R}^2$ elipsu se středem v počátku, poloosy jsou ve směru souřadných os $x_1, x_2$ a mají délky $a$ resp. $b$.

Nyní uvažme rovnici $x^T Ax = 1$, kde $A \in \mathbb{R}^{n \times n}$ je positivně definitní a $x = (x_1, \ldots, x_n)^T$ je vektor proměnných. Nahlédneme, že rovnice popisuje elipsoid v prostoru $\mathbb{R}^n$. Buď $A = Q\Lambda Q^T$ spektrální rozklad. Při substituci $y \coloneqq Q^T x$ dostaneme

$$1 = x^T Ax = x^T Q \Lambda Q^T x = y^T \Lambda y = \sum_{i=1}^n \lambda_i y_i^2 = \sum_{i=1}^n \frac{1}{(\lambda_i^{-1/2})^2} y_i^2.$$

Dostáváme tedy popis elipsoidu se středem v počátku, poloosy jsou ve směru souřadnic a mají délky $\frac{1}{\sqrt{\lambda_1}}, \ldots, \frac{1}{\sqrt{\lambda_n}}$. Nicméně, tento popis je v prostoru po transformaci $y = Q^T x$. Vrátíme zpět transformací $x = Qy$. Protože $Q$ je ortogonální matice, dostaneme stejný elipsoid se středem v počátku, jen nějak pootočený či překlopený. Protože kanonická báze $e_1, \ldots, e_n$ se zobrazí na sloupce matice $Q$ (což jsou vlastní vektory matice $A$), tak poloosy původního elipsoidu budou ve směrech vlastních vektorů matice $A$.

Je-li $A$ symetrická, ale ne positivně definitní, analýza bude stejná. Jenom nedostaneme elipsoid, ale jiný geometrický útvar (hyperboloid aj.)

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 12.23 — Některé kvadriky v $\mathbb{R}^3$)</span></p>

Následující obrázky znázorňují některé další třídimenziální kvadriky:

- **Elipsoid:** $\frac{x_1^2}{a^2} + \frac{x_2^2}{b^2} + \frac{x_3^2}{c^2} = 1$
- **Kuželová plocha:** $\frac{x_1^2}{a^2} + \frac{x_2^2}{b^2} - \frac{x_3^2}{c^2} = 0$
- **Hyperbolický paraboloid:** $\frac{x_1^2}{a^2} - \frac{x_2^2}{b^2} - x_3 = 0$
- **Hyperbolická válcová plocha:** $\frac{x_1^2}{a^2} - \frac{x_2^2}{b^2} = 1$
- **Jednodílný hyperboloid:** $\frac{x_1^2}{a^2} + \frac{x_2^2}{b^2} - \frac{x_3^2}{c^2} = 1$
- **Dvojdílný hyperboloid:** $-\frac{x_1^2}{a^2} - \frac{x_2^2}{b^2} + \frac{x_3^2}{c^2} = 1$

</div>

### Shrnutí ke kapitole 12

Kvadratickou formou v prostoru $\mathbb{R}^n$ je polynom $f(x) = x^T Ax$, kde $A \in \mathbb{R}^{n \times n}$ je symetrická. U obecných prostorů pracujeme v souřadnicích podobně jako u lineárních zobrazení. Změníme-li souřadný systém, matice se změní na matici $S^T AS$, kde $S$ je matice přechodu mezi souřadnými systémy. Ústřední věta této kapitoly, Sylvestrův zákon setrvačnosti, pak tvrdí, že vždycky můžeme najít souřadný systém, ve kterém je matice diagonální. Navíc má tato diagonální matice pro pevnou kvadratickou formu vždy stejnou signaturu — stejný počet kladných a záporných čísel (to je hlavní poselství věty!). To nám umožňuje kvadratické formy jednoduše klasifikovat a popisovat. Signaturu pro danou matici určíme jednoduše pomocí elementárních úprav aplikovaných jak na řádky, tak symetricky i na sloupce. Tím získáme mimochodem efektivní metodu na testování positivní (semi-)definitnosti aj. Teorie kvadratických forem nám také dává účinný nástroj jak analyzovat polynomiální rovnice typu $x^T Ax + b^T x + c = 0$ a jak charakterizovat různé kuželosečky a kvadriky, jako jsou například elipsoidy.

Kvadratické formy úzce souvisí s bilineárními formami. Reálný skalární součin je vždy bilineární formou. Speciálně v prostoru $\mathbb{R}^n$ mají bilineární formy tvar $b(x, y) = x^T Ay$. V obecných prostorech mají podobné vyjádření, ale v řeči souřadnic. Konkrétně, $b(x, y) = [x]_B^T A [y]_B$ pro danou bázi $B$.

## Kapitola 13 — Maticové rozklady

Top 10 algoritmů 20. století podle [Dongarra and Sullivan, 2000; Cipra, 2000]:

1. *Metoda Monte Carlo* (1946, J. von Neumann, S. Ulam, a N. Metropolis) — pomocí simulací s náhodnými čísly spočítáme přibližná řešení problémů, které jsou velmi těžké na to spočítat jejich řešení přesně.
2. *Simplexová metoda pro lineární programování* (1946, G. Dantzig) — metoda na výpočet optimalizačních úloh s lineárním kriteriem i omezeními.
3. *Iterační metody Krylovovských podprostorů* (1950, M. Hestenes, E. Stiefel, C. Lanczos) — metody na řešení velkých a řídkých soustav lineárních rovnic.
4. *Dekompozice matic* (1951, A. Householder) — maticové rozklady jako je např. Choleského rozklad, LU rozklad, QR rozklad, spektrální rozklad, Schurova triangularizace nebo SVD rozklad.
5. *Překladač Fortranu* (1957, J. Backus) — programovací jazyk Fortran, který jako jeden z prvních umožnil náročné numerické výpočty.
6. *QR algoritmus* (1961, J. Francis) — algoritmus pro výpočet vlastních čísel.
7. *Quicksort* (1962, A. Hoare) — prakticky rychlý algoritmus na třídění prvků.
8. *Rychlá Fourierova transformace* (1965, J. Cooley, J. Tukey) — pro rychlé násobení polynomů a čísel, zpracování signálu a mnoho dalších věcí.
9. *Algoritmus „Integer relation detection"* (1977, H. Ferguson, R. Forcade) — zobecnění Euklidova algoritmu postupného dělení na problém: Dáno $n$ reálných čísel $x_1, \ldots, x_n$, existuje netriviální celočíselná kombinace $a_1 x_1 + \ldots + a_n x_n = 0$?
10. *Metoda více pólů — „Fast multipole algorithm"* (1987, L. Greengard, V. Rokhlin) — simulace v problému výpočtu sil dalekého dosahu v úloze $n$ těles. Seskupuje zdroje ležící u sebe a pracuje s nimi jako s jediným.

Vidíme, že maticové rozklady (dekompozice) byly do seznamu zařazeny. S několika rozklady jsme se již setkali (LU rozklad, spektrální rozklad, Choleského rozklad, ...). QR rozklad, kterému se budeme věnovat v sekci 13.2, se v seznamu objevuje skrytě ještě jednou, protože je základem QR algoritmu. Jeho důležitost je tedy patrná.

V této kapitole budeme uvažovat standardní skalární součin v $\mathbb{R}^n$ a eukleidovskou normu (pokud není explicitně řečeno jinak).

### 13.1 Householderova transformace

Motivace k této sekci je následující. Mějme dány dva vektory $x, y \in \mathbb{R}^n$ a chceme najít lineární zobrazení $x \mapsto Hx$ takové, které zobrazí vektor $x$ na vektor $y$. Aby mělo lineární zobrazení pěkné vlastnosti, požadujeme navíc, aby matice $H$ byla ortogonální. Protože takové lineární zobrazení zachovává délky (věta 8.66), musí nutně mít oba vektory $x, y$ stejnou normu.

Takovéto zobrazení vždy existuje, a za matici $H$ lze zvolit vhodnou Householderovu matici. Připomeňme (příklad 8.65), že Householderova matice je definována jako $H(x) \coloneqq I_n - \frac{2}{x^T x} x x^T$, kde $o \neq x \in \mathbb{R}^n$. Tato matice je ortogonální a symetrická. Vhodnou volbou vektorů $x, y$ pak může nahradit elementární matice při výpočtu odstupňovaného tvaru matice. Tento postup se nazývá Householderova transformace.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Věta 13.1 — Householderova transformace)</span></p>

Pro každé $x, y \in \mathbb{R}^n$, $x \neq y$, $\|x\|_2 = \|y\|_2$ platí $y = H(x - y)x$.

</div>

*Důkaz.* Počítejme

$$H(x - y)x = \left(I_n - \frac{2}{(x - y)^T(x - y)}(x - y)(x - y)^T\right) x =$$

$$= x - \frac{2(x - y)^T x}{(x - y)^T(x - y)}(x - y) = x - \frac{2\|x\|_2^2 - 2y^T x}{(x - y)^T(x - y)}(x - y) =$$

$$= x - \frac{\|x\|_2^2 + \|y\|_2^2 - 2y^T x}{\|x - y\|_2^2}(x - y) = x - \frac{\|x - y\|_2^2}{\|x - y\|_2^2}(x - y) = x - (x - y) = y.$$

Householderova matice tedy převádí jeden vybraný vektor $x$ na jiný $y$ se stejnou normou tím, že vynásobíme vektor $x$ zleva vhodnou Householderovou maticí. Speciálně lze převést každý vektor na vhodný násobek jednotkového vektoru:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(Důsledek 13.2)</span></p>

Buď $x \in \mathbb{R}^n$ a definujme

$$H \coloneqq \begin{cases} H(x - \|x\|_2 e_1), & \text{pokud } x \neq \|x\|_2 e_1, \\ I_n, & \text{jinak.} \end{cases}$$

Potom $Hx = \|x\|_2 e_1$.

</div>

*Důkaz.* Případ $x = \|x\|_2 e_1$ je jasný. Jinak použijeme větu 13.1; vektory $x$, $\|x\|_2 e_1$ mají stejnou normu.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 13.3)</span></p>

Buď $x = (2, 2, 1)^T$. Pak $\|x\|_2 = 3$ a tedy

$$H = H(x - 3e_1) = \frac{1}{3}\begin{pmatrix} 2 & 2 & 1 \\ 2 & -1 & -2 \\ 1 & -2 & 2 \end{pmatrix}.$$

Nyní $Hx = (3, 0, 0)^T$, tedy lineární zobrazení $x \mapsto Hx$ zobrazuje vektor $x = (2, 2, 1)^T$ na násobek jednotkového vektoru.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 13.4 — Doplnění na ortonormální bázi)</span></p>

Buď $u \in \mathbb{R}^n$ vektor jednotkové délky, tj. $\|u\|_2 = 1$. Sestrojte ortogonální matici $Q$ s prvním sloupcem rovným $u$. Jinými slovy, doplňte $u$ na ortonormální bázi prostoru $\mathbb{R}^n$.

*Řešení:* Je-li $u = e_1$, pak zřejmě stačí volit $Q = I_n$. V opačném případě můžeme zvolit matici $Q \coloneqq H(e_1 - u)$, tedy Householderovu matici pro vektor $e_1 - u$. Zdůvodnění je jednoduché, první sloupec této matice je podle Householderovy transformace roven $Qe_1 = H(e_1 - u)e_1 = u$.

</div>

Mějme matici $A \in \mathbb{R}^{m \times n}$. Householderovu matici $H$ sestrojíme tak, aby první sloupec matice $A$ převedla Householderova transformace podle důsledku 13.2 na násobek $e_1$. Vynásobením $HA$ tak vynulujeme prvky v prvním sloupci $A$ až na první z nich. Rekurzivním voláním transformace pak převedeme matici do odstupňovaného tvaru. Tento postup je tedy alternativou k elementárním řádkovým úpravám. Máme tu však ještě něco navíc, a to tzv. QR rozklad, o němž hovoříme podrobněji v následující sekci.

Podobně lze použít i Givensovy matice (problém 13.2). Vynásobením matice $A$ vhodnou Givensovou maticí zleva dokážeme vynulovat libovolný (ale pouze jeden) prvek pod pivotem; toto je společná vlastnost s maticí elementární úpravy. Abychom vynulovali všechny prvky pod pivotem, musíme tedy použít příslušné Givensovy matice vícekrát. Podrobně se však Givensovými maticemi zabývat nebudeme.

### 13.2 QR rozklad

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Věta 13.5 — QR rozklad)</span></p>

Pro každou matici $A \in \mathbb{R}^{m \times n}$ existuje ortogonální $Q \in \mathbb{R}^{m \times m}$ a horní trojúhelníková matice $R \in \mathbb{R}^{m \times n}$ s nezápornou diagonálou tak, že $A = QR$.

</div>

*Důkaz.* Matematickou indukcí podle $n$, tj. počtu sloupců. Je-li $n = 1$, pak $A = a \in \mathbb{R}^m$ a pro matici $H$ sestrojenou podle důsledku 13.2 platí $Ha = \|a\|_2 e_1$. Stačí položit $Q \coloneqq H^T$ a $R \coloneqq \|a\|_2 e_1$.

Indukční krok $n \leftarrow n - 1$. Aplikací důsledku 13.2 na první sloupec matice $A$ dostaneme $HA_{*1} = \|A_{*1}\|_2 e_1$. Tedy $HA$ je tvaru

$$HA = \begin{pmatrix} \alpha & b^T \\ o & B \end{pmatrix},$$

kde $B \in \mathbb{R}^{(m-1) \times (n-1)}$ a $\alpha = \|A_{*1}\|_2 \ge 0$. Podle indukčního předpokladu existuje rozklad $B = Q'R'$, kde $Q' \in \mathbb{R}^{(m-1) \times (m-1)}$ je ortogonální a $R' \in \mathbb{R}^{(m-1) \times (n-1)}$ horní trojúhelníková s nezápornou diagonálou. Upravme

$$\begin{pmatrix} 1 & o^T \\ o & Q'^T \end{pmatrix} HA = \begin{pmatrix} 1 & o^T \\ o & Q'^T \end{pmatrix} \begin{pmatrix} \alpha & b^T \\ o & B \end{pmatrix} = \begin{pmatrix} \alpha & b^T \\ o & R' \end{pmatrix}.$$

Označme

$$Q \coloneqq H^T \begin{pmatrix} 1 & o^T \\ o & Q' \end{pmatrix}, \quad R \coloneqq \begin{pmatrix} \alpha & b^T \\ o & R' \end{pmatrix}.$$

Matice $Q$ je ortogonální a $R$ je horní trojúhelníková s nezápornou diagonálou. Nyní rovnice (13.1) má tvar $Q^T A = R$, neboli $A = QR$ je hledaný rozklad.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Algoritmus 13.6 — QR rozklad)</span></p>

**Vstup:** matice $A \in \mathbb{R}^{m \times n}$.

1. $Q \coloneqq I_m$, $R \coloneqq A$,
2. **for** $j \coloneqq 1$ **to** $\min(m, n)$ **do**
3. &emsp; $x \coloneqq R(j : m, j)$,
4. &emsp; **if** $x \neq \|x\|_2 e_1$ **then**
5. &emsp;&emsp; $x \coloneqq x - \|x\|_2 e_1$,
6. &emsp;&emsp; $H(x) \coloneqq I_{m-j+1} - \frac{2}{x^T x} x x^T$,
7. &emsp;&emsp; $H \coloneqq \begin{pmatrix} I_{j-1} & 0 \\ 0 & H(x) \end{pmatrix}$,
8. &emsp;&emsp; $R \coloneqq HR$, $Q \coloneqq QH$,
9. &emsp; **end if**
10. **end for**

**Výstup:** matice $Q, R$ z QR rozkladu matice $A$ (platí $A = QR$).

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 13.7 — QR rozklad)</span></p>

Buď

$$A = \begin{pmatrix} 0 & -20 & -14 \\ 3 & 27 & -4 \\ 4 & 11 & -2 \end{pmatrix}.$$

**První iterace:**

$x = A_{*1} - \|A_{*1}\|e_1 = (-5, 3, 4)^T$,

$$Q_1 = I_3 - 2\frac{xx^T}{x^T x} = \frac{1}{25}\begin{pmatrix} 0 & 15 & 20 \\ 15 & 16 & -12 \\ 20 & -12 & 9 \end{pmatrix}, \quad Q_1 A = \begin{pmatrix} 5 & 25 & -4 \\ 0 & 0 & -10 \\ 0 & -25 & -10 \end{pmatrix}.$$

**Druhá iterace:**

$x = (0, -25)^T - 25e_1 = (-25, -25)^T$,

$$Q_2 = I_2 - 2\frac{xx^T}{x^T x} = \begin{pmatrix} 0 & -1 \\ -1 & 0 \end{pmatrix}, \quad \tilde{Q}_2 = \begin{pmatrix} 25 & 10 \\ 0 & 10 \end{pmatrix}.$$

**Výsledek:**

$$Q = Q_1 \begin{pmatrix} 1 & 0 \\ 0 & Q_2 \end{pmatrix} = \frac{1}{25}\begin{pmatrix} 0 & -20 & -15 \\ 15 & 12 & -16 \\ 20 & -9 & 12 \end{pmatrix}, \quad R = \begin{pmatrix} 5 & 25 & -4 \\ 0 & 25 & 10 \\ 0 & 0 & 10 \end{pmatrix}.$$

</div>

QR rozklad je jednoznačný jen za určitých předpokladů. Například pro nulovou matici $A = 0$ je $R = 0$ a $Q$ libovolná ortogonální matice, tedy jednoznačnost tu nenastává.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Tvrzení 13.8 — Jednoznačnost QR rozkladu)</span></p>

Pro regulární matici $A \in \mathbb{R}^{n \times n}$ je QR rozklad jednoznačný a matice $R$ má na diagonále kladné hodnoty.

</div>

*Důkaz.* Ze vztahu $A = QR$ plyne, že $R$ je regulární, a proto musí mít nenulovou, tudíž kladnou, diagonálu. Jednoznačnost ukážeme sporem. Nechť $A$ má dva různé rozklady $A = Q_1 R_1 = Q_2 R_2$. Pak $Q_2^T Q_1 = R_2 R_1^{-1}$, a tuto matici označíme jako $U$. Zřejmě $U$ je ortogonální (je to součin ortogonálních matic $Q_2^T$ a $Q_1$) a horní trojúhelníková (je to součin horních trojúhelníkových matic $R_2$ a $R_1^{-1}$). Speciálně, $U$ má tvar $U_{*1} = (u_{11}, 0, \ldots, 0)^T$, kde $u_{11} > 0$. Aby měl jednotkovou velikost, musí $u_{11} = 1$ a proto $U_{*1} = e_1$. Druhý sloupec je kolmý na první, proto $u_{21} = 0$, a aby měl jednotkovou velikost, musí $u_{22} = 1$. Tedy $U_{*2} = e_2$. Atd. pokračujeme dále až dostaneme $U = I_n$, z čehož $Q_1 = Q_2$ a $R_1 = R_2$. To je spor.

Věta se dá zobecnit i na případ kdy $A \in \mathbb{R}^{m \times n}$ má lineárně nezávislé sloupce. Pak matice $R$ a prvních $n$ sloupců $Q$ je jednoznačně určeno, a diagonála matice $R$ je kladná.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Poznámka 13.9 — QR rozklad pomocí Gramovy–Schmidtovy ortogonalizace)</span></p>

QR rozklad matice $A \in \mathbb{R}^{m \times n}$ lze sestrojit i pomocí Gramovy–Schmidtovy ortogonalizace. Prakticky se to sice nedělá, protože právě díky použití ortogonálních matic je Householderova transformace numericky lepší, ale umožní nám to lépe pochopit vztah obou metod.

Základní myšlenka je následující: Zatímco Householderovými transformacemi (tj., sekvencí ortogonálních matic) upravujeme matici $A$ na horní trojúhelníkovou, Gramova–Schmidtova ortogonalizace funguje přesně naopak — pomocí sekvence vhodných horních trojúhelníkových matic upravíme $A$ na matici s ortonormálními sloupci.

Konkrétně popíšeme Gramovu–Schmidtovu ortogonalizaci takto. Mějme matici $A \in \mathbb{R}^{m \times n}$, jejíž sloupce chceme zortonormalizovat. Budeme postupovat podle algoritmu 8.23 s tím, že vektory $x_1, \ldots, x_n$ jsou sloupce matice $A$ a v průběhu výpočtu budeme sloupce matice $A$ nahrazovat vektory $y_k$ a $z_k$. Krok 2 algoritmu, který má tvar

$$y_k \coloneqq x_k - \sum_{j=1}^{k-1} \langle x_k, z_j \rangle z_j,$$

vyjádříme maticově tak, že matici $A$ vynásobíme zprava maticí

$$\begin{pmatrix} 1 & & \alpha_1 \\ & \ddots & \vdots \\ & & 1 & \alpha_{k-1} \\ & & & 1 \\ & & & & 1 \end{pmatrix},$$

kde $\alpha_j = -\langle x_k, z_j \rangle$, $j = 1, \ldots, k - 1$. Podobně krok 3, který má tvar $z_k \coloneqq \frac{1}{\|y_k\|} y_k$, vyjádříme vynásobením matice $A$ zprava diagonální maticí s prvky $1, \ldots, 1, \frac{1}{\|y_k\|}, 1, \ldots, 1$ na diagonále. Protože obě matice, kterými jsme násobili $A$ zprava, jsou horní trojúhelníkové, můžeme celou ortogonalizaci vyjádřit jako

$$AR_1 \ldots R_\ell = Q,$$

kde $R_1, \ldots, R_\ell$ jsou horní trojúhelníkové matice a $Q$ má ortonormální sloupce. Hledaný QR rozklad nyní dostaneme, kde $R \coloneqq (R_1 \ldots R_\ell)^{-1}$ je rovněž horní trojúhelníková matice.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Poznámka 13.10 — QR rozklad pomocí Choleského rozkladu)</span></p>

Buď $A \in \mathbb{R}^{m \times n}$ hodnosti $n$ a $A = QR$ její hledaný QR rozklad. Vyjádříme matice $Q, R$ blokově jako

$$A = QR = (\tilde{Q} \quad \tilde{Q}') \begin{pmatrix} \tilde{R} \\ 0 \end{pmatrix} = \tilde{Q}\tilde{R},$$

kde $\tilde{Q} \in \mathbb{R}^{m \times n}$ tvoří prvních $n$ sloupců matice $Q$ a $\tilde{R}$ tvoří prvních $n$ řádků matice $R$. Matice $\tilde{R}$ je regulární horní trojúhelníková matice s kladnou diagonálou. Pak $A^T A = R^T Q^T QR = R^T R = \tilde{R}^T \tilde{R}$. Matici $\tilde{R}$ tedy můžeme sestrojit Choleského rozkladem z matice vzniklé součinem $A^T A$. Z rovnice $A = \tilde{Q}\tilde{R}$ pak jednoduše vyjádříme $\tilde{Q} = A\tilde{R}^{-1}$. Zbylé sloupečky matice $Q$ dopočítáme libovolně tak, aby matice $Q$ byla ortogonální (že to lze máme zaručeno důsledkem 8.27).

</div>

### 13.3 Aplikace QR rozkladu

QR rozklad se dá použít na řešení mnoha úloh, se kterými jsme se doposud setkali. Jeho hlavní výhodou je, že pracuje s ortogonální maticí $Q$. Protože ortogonální matice zachovávají normu (věta 8.66(2)), tak se zaokrouhlovací chyby příliš nezvětšují. To je důvod, proč se ortogonální matice hojně využívají v numerických metodách.

#### QR rozklad a soustavy rovnic

Uvažujme soustavu lineárních rovnic $Ax = b$, kde $A \in \mathbb{R}^{n \times n}$ je regulární. Řešení vypočítáme následujícím způsobem: Vypočítej QR rozklad $A = QR$. Pak soustava má tvar $QRx = b$, neboli $Rx = Q^T b$. Protože $R$ je horní trojúhelníková matice, řešení dostaneme snadno zpětnou substitucí.

Asymptotická složitost QR rozkladu, a tedy i vyřešení soustavy rovnic, je $\frac{4}{3}n^3$. Oproti Gaussově eliminaci je tedy tento způsob přibližně dvakrát pomalejší (viz poznámka 2.19), avšak je numericky stabilnější a přesnější.

#### QR rozklad a ortogonalizace

Pro následující si nejprve zavedeme tzv. *redukovaný QR rozklad*. Nechť $A \in \mathbb{R}^{m \times n}$ má lineárně nezávislé sloupce. Pak QR rozklad rozepíšeme blokově

$$A = QR = (\tilde{Q} \quad \tilde{Q}') \begin{pmatrix} \tilde{R} \\ 0 \end{pmatrix} = \tilde{Q}\tilde{R},$$

kde $\tilde{Q} \in \mathbb{R}^{m \times n}$ tvoří prvních $n$ sloupců matice $Q$ a $\tilde{R}$ tvoří prvních $n$ řádků matice $R$. Matice $\tilde{R}$ je regulární.

Nyní se podíváme, jak QR rozklad aplikovat k nalezení ortonormální báze daného prostoru; je to tedy alternativa ke Gramově–Schmidtově ortogonalizaci v $\mathbb{R}^m$. Nechť $A \in \mathbb{R}^{m \times n}$ má lineárně nezávislé sloupce a chceme sestrojit ortonormální bázi sloupcového prostoru $\mathcal{S}(A)$. Z rovnosti $A = \tilde{Q}\tilde{R}$ a regularity $\tilde{R}$ vyplývá (tvrzení 5.66), že $\mathcal{S}(A) = \mathcal{S}(\tilde{Q})$. Tedy ortonormální bázi $\mathcal{S}(A)$ tvoří sloupce $\tilde{Q}$.

Vzhledem k vlastnostem ortogonálních matic pak $\tilde{Q}'$ tvoří ortonormální bázi $\operatorname{Ker}(A^T)$, protože $\operatorname{Ker}(A^T)$ je ortogonální doplněk $\mathcal{S}(A)$. Z QR rozkladu matice $A$ resp. $A^T$ tedy dokážeme vyčíst ortonormální bázi všech základních maticových prostorů — řádkového, sloupcového a jádra.

#### QR rozklad a rozšíření na ortonormální bázi

Buď $a \in \mathbb{R}^n$, $\|a\|_2 = 1$, a buď $a = Qr$ jeho QR rozklad. Aby $r$ byl vektor $r$, bráno jako matice s jedním sloupcem, v horním trojúhelníkovém tvaru s nezápornou diagonálou, musí mít tvar $r = (\alpha, 0, \ldots, 0)^T$ pro nějaké $\alpha \ge 0$. Protože $\|a\|_2 = 1$ a $Q$ je ortogonální, musí $\|r\|_2 = 1$. Tudíž $r = e_1$, z čehož $a = Q_{*1}$. Vektor $a$ tak leží v prvním sloupci $Q$ a ostatní sloupce představují jeho rozšíření na ortonormální bázi.

#### QR rozklad a projekce do podprostoru

Nechť $A \in \mathbb{R}^{m \times n}$ má lineárně nezávislé sloupce. Víme (věta 8.49), že projekce vektoru $x \in \mathbb{R}^m$ do sloupcového prostoru $\mathcal{S}(A)$ je tvaru $x' = A(A^T A)^{-1}A^T x$. Výraz můžeme zjednodušit s použitím redukovaného QR rozkladu $A = \tilde{Q}\tilde{R}$. Protože $\mathcal{S}(A) = \mathcal{S}(\tilde{Q})$, hledáme projekci do prostoru s ortonormální bází danou ve sloupcích matice $\tilde{Q}$. Podle poznámky 8.52 je matice projekce $\tilde{Q}\tilde{Q}^T$ a vektor $x$ se projektuje na vektor $x' = \tilde{Q}\tilde{Q}^T x$.

#### QR rozklad a metoda nejmenších čtverců

Metoda nejmenších čtverců (sekce 8.5) spočívá v přibližném řešení přeurčené soustavy rovnic $Ax = b$, kde $A \in \mathbb{R}^{m \times n}$, $m > n$. Nechť $A$ má hodnost $n$, pak přibližné řešení metodou nejmenších čtverců je

$$x^* = (A^T A)^{-1}A^T b = \tilde{R}^{-1}(\tilde{R}^T)^{-1}\tilde{R}^T \tilde{Q}^T b = \tilde{R}^{-1}\tilde{Q}^T b.$$

Jinými slovy, $x^*$ získáme jako řešení regulární soustavy $\tilde{R}x = \tilde{Q}^T b$, a to zpětnou substitucí protože matice $\tilde{R}$ je horní trojúhelníková. Povšimněme si analogie s řešením regulární soustavy $Ax = b$, které vedlo na $Rx = Q^T b$; nyní máme oříznutou soustavu $\tilde{R}x = \tilde{Q}^T b$.

#### QR algoritmus

*QR algoritmus* je metoda na výpočet vlastních čísel matice $A \in \mathbb{R}^{n \times n}$, která se stala základem soudobých efektivních metod.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Algoritmus 13.11 — QR algoritmus)</span></p>

**Vstup:** matice $A \in \mathbb{R}^{n \times n}$.

1. $A_0 \coloneqq A$, $i \coloneqq 0$,
2. **while not** splněna ukončovací podmínka **do**
3. &emsp; sestroj QR rozklad matice $A_i$, tj. $A_i = QR$,
4. &emsp; $A_{i+1} \coloneqq RQ$,
5. &emsp; $i \coloneqq i + 1$,
6. **end while**

**Výstup:** matice $A_i$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Tvrzení 13.12)</span></p>

Matice $A_0, A_1, \ldots$ jsou si navzájem podobné.

</div>

*Důkaz.* $A_{i+1} = RQ = I_n RQ = Q^T QRQ = Q^T A_i Q$.

Matice $A_i$ na výstupu je podobná s $A$, a má tím pádem i stejná vlastní čísla. Jak je zjistíme? Algoritmus vesměs konverguje (případy kdy nekonverguje jsou řídké, skoro umělé; dlouho nebyl znám případ kdy nekonvergoval) k blokově horní trojúhelníkové matici s bloky o velikosti 1 a 2. Bloky o velikosti 1 jsou vlastní čísla, a z bloků o velikosti 2 jednoduše dopočítáme dvojice komplexně sdružených vlastních čísel.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 13.13 — Iterace QR algoritmu)</span></p>

Iterace QR algoritmu pro danou matici $A$:

$$A = \begin{pmatrix} 2 & 4 & 2 \\ 4 & 2 & 2 \\ 2 & 2 & -1 \end{pmatrix} \to \begin{pmatrix} 6.1667 & -2.4623 & 0.8616 \\ -2.4623 & -1.2576 & -0.2598 \\ 0.8616 & -0.2598 & -1.9091 \end{pmatrix} \to$$

$$\to \begin{pmatrix} 6.9257 & 0.7725 & 0.2586 \\ 0.7725 & -1.9331 & 0.0224 \\ 0.2586 & 0.0224 & -1.9925 \end{pmatrix} \to \begin{pmatrix} 6.9939 & -0.2225 & 0.0742 \\ -0.2225 & -1.9945 & -0.0018 \\ 0.0742 & -0.0018 & -1.9994 \end{pmatrix} \to$$

$$\to \begin{pmatrix} 6.9995 & 0.0636 & 0.0212 \\ 0.0636 & -1.9996 & 0.0001 \\ 0.0212 & 0.0001 & -1.9999 \end{pmatrix} \to \begin{pmatrix} 7.0000 & -0.0182 & 0.0061 \\ -0.0182 & -2.0000 & -10^{-5} \\ 0.0061 & -10^{-5} & -2.0000 \end{pmatrix}.$$

Symetrická matice konverguje k diagonální. Přesnost vypočítaných vlastních čísel určuje věta 10.58 o Geršgorinových discích.

</div>

### 13.4 SVD rozklad

Stejně jako QR rozklad je SVD rozklad jednou ze základních technik v numerických výpočtech.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Věta 13.14 — SVD rozklad)</span></p>

Buď $A \in \mathbb{R}^{m \times n}$, $q \coloneqq \min\lbrace m, n \rbrace$. Pak existuje diagonální matice $\Sigma \in \mathbb{R}^{m \times n}$ s prvky $\sigma_{11} \ge \ldots \ge \sigma_{qq} \ge 0$ a ortogonální matice $U \in \mathbb{R}^{m \times m}$, $V \in \mathbb{R}^{n \times n}$ takové, že $A = U\Sigma V^T$.

</div>

Ideu důkazu uvádíme za algoritmem 13.17, který konstruuje SVD rozklad. Kladným číslům na diagonále $\sigma_{11}, \ldots, \sigma_{rr}$ říkáme *singulární čísla* matice $A$ a značíme je obvykle $\sigma_1, \ldots, \sigma_r$. Zjevně $r = \operatorname{rank}(A)$. Singulární čísla jsou jednoznačná, ale matice $U, V$, a tím pádem i SVD rozklad, být nemusí.

Transpozice ortogonální matice je opět ortogonální matice, proto se zdá na první pohled nesmyslné transponovat matici $V$ v rozkladu $A = U\Sigma V^T$. Důvod pro to je spíš zvyklost. Takto můžeme najít báze určitých prostorů ve sloupcích matic $U, V$ (viz věta 13.20), jinak bez transpozice by to byly sloupce $U$ a řádky $V$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Věta 13.15 — Vztah singulárních a vlastních čísel)</span></p>

Buď $A \in \mathbb{R}^{m \times n}$, $r = \operatorname{rank}(A)$, a nechť $A^T A$ má vlastní čísla $\lambda_1 \ge \ldots \ge \lambda_n$. Pak singulární čísla matice $A$ jsou $\sigma_i = \sqrt{\lambda_i}$, $i = 1, \ldots, r$.

</div>

*Důkaz.* Nechť $A = U\Sigma V^T$ je SVD rozklad matice $A$. Pak

$$A^T A = V\Sigma^T U^T U \Sigma V^T = V\Sigma^T \Sigma V^T = V \operatorname{diag}(\sigma_1^2, \ldots, \sigma_q^2, 0, \ldots, 0)V^T,$$

což je spektrální rozklad positivně semidefinitní matice $A^T A$. Proto určující prvky diagonální matice jsou její vlastní čísla, čili $\lambda_i = \sigma_i^2$.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 13.16)</span></p>

Buď $Q \in \mathbb{R}^{n \times n}$ ortogonální. Pak $Q^T Q = I_n$ má vlastní čísla samé jedničky. Tedy ortogonální matice $Q$ má singulární čísla také samé jedničky.

</div>

Důkaz věty 13.15 prozradil navíc, že matice $V$ je ortogonální maticí ze spektrálního rozkladu $A^T A$. Podobně, matice $U$ je ortogonální maticí ze spektrálního rozkladu $AA^T$:

$$AA^T = U\Sigma V^T V\Sigma^T U^T = U\Sigma\Sigma^T U^T = U \operatorname{diag}(\sigma_1^2, \ldots, \sigma_q^2, 0, \ldots, 0)U^T.$$

Bohužel, spektrální rozklady matic $A^T A$ a $AA^T$ nemůžeme použít ke konstrukci SVD rozkladu, protože nejsou jednoznačné. Použít můžeme jen jeden a druhý dopočítat trochu jinak.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Algoritmus 13.17 — SVD rozklad)</span></p>

**Vstup:** matice $A \in \mathbb{R}^{m \times n}$.

1. Sestroj $V\Lambda V^T$ spektrální rozklad matice $A^T A$;
2. $r \coloneqq \operatorname{rank}(A)$;
3. $\sigma_i \coloneqq \sqrt{\lambda_i}$, $i = 1, \ldots, r$;
4. $S \coloneqq \operatorname{diag}(\sigma_1, \ldots, \sigma_r)$, $\Sigma \coloneqq \begin{pmatrix} S & 0 \\ 0 & 0 \end{pmatrix}$;
5. buď $V_1$ matice tvořená prvními $r$ sloupci $V$;
6. $U_1 \coloneqq AV_1 S^{-1}$;
7. doplň $U_1$ na ortogonální matici $U = (U_1 \mid U_2)$;

**Výstup:** matice $U, \Sigma, V^T$ z SVD rozkladu matice $A$ (platí $A = U\Sigma V^T$).

</div>

*Důkaz.* Z věty 13.15 víme, že $\sigma_1, \ldots, \sigma_r$ jsou hledaná singulární čísla a zjevně $V$ je ortogonální. Musíme dokázat, že $U_1$ má ortonormální sloupce a $A = U\Sigma V^T$.

Z rovnosti $A^T A = V\Lambda V^T$ odvodíme $\Lambda = V^T A^T A V$ a odříznutím posledních $n - r$ řádků a sloupců dostaneme matici $\operatorname{diag}(\lambda_1, \ldots, \lambda_r) = V_1^T A^T A V_1$. Nyní je vidět, že $U_1$ má ortonormální sloupce, neboť

$$U_1^T U_1 = (S^{-1})^T V_1^T A^T A V_1 S^{-1} = (S^{-1})^T S^2 S^{-1} = I_r.$$

Zbývá ukázat, že $A = U\Sigma V^T$, neboli $\Sigma = U^T AV$. Rozložme $V = (V_1 \mid V_2)$. Odříznutím prvních $r$ řádků a sloupců v matici $\Lambda = V^T A^T AV$ dostaneme $0 = V_2^T A^T AV_2$, z čehož $AV_2 = 0$ (důsledek 8.47(1)). Nyní s využitím rovnosti $AV_1 = U_1 S$ máme

$$U^T AV = U^T A(V_1 \mid V_2) = (U^T U_1 S \mid U^T AV_2) = \begin{pmatrix} S & 0 \\ 0 & 0 \end{pmatrix} = \Sigma.$$

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 13.18 — SVD rozklad)</span></p>

Mějme

$$A = \begin{pmatrix} 1 & 1 \\ 2 & 0 \\ 0 & -2 \end{pmatrix}.$$

Spektrální rozklad matice $A^T A$:

$$A^T A = \begin{pmatrix} \frac{\sqrt{2}}{2} & \frac{\sqrt{2}}{2} \\ \frac{\sqrt{2}}{2} & -\frac{\sqrt{2}}{2} \end{pmatrix} \begin{pmatrix} 6 & 0 \\ 0 & 4 \end{pmatrix} \begin{pmatrix} \frac{\sqrt{2}}{2} & \frac{\sqrt{2}}{2} \\ \frac{\sqrt{2}}{2} & -\frac{\sqrt{2}}{2} \end{pmatrix} \equiv V\Lambda V^T.$$

Určení $S$: $S = \begin{pmatrix} \sqrt{6} & 0 \\ 0 & 2 \end{pmatrix}$.

Určení $U_1$ (v tomto příkladu máme $V_1 = V$):

$$U_1 = AV_1 S^{-1} = \begin{pmatrix} \frac{\sqrt{3}}{3} & 0 \\ \frac{\sqrt{3}}{3} & \frac{\sqrt{2}}{2} \\ -\frac{\sqrt{3}}{3} & \frac{\sqrt{2}}{2} \end{pmatrix}.$$

Doplnění $U_1$ ortogonální maticí $U$:

$$U = \begin{pmatrix} \frac{\sqrt{3}}{3} & 0 & -\frac{\sqrt{6}}{3} \\ \frac{\sqrt{3}}{3} & \frac{\sqrt{2}}{2} & \frac{\sqrt{6}}{6} \\ -\frac{\sqrt{3}}{3} & \frac{\sqrt{2}}{2} & -\frac{\sqrt{6}}{6} \end{pmatrix}.$$

Výsledný SVD rozklad:

$$A = \begin{pmatrix} \frac{\sqrt{3}}{3} & 0 & -\frac{\sqrt{6}}{3} \\ \frac{\sqrt{3}}{3} & \frac{\sqrt{2}}{2} & \frac{\sqrt{6}}{6} \\ -\frac{\sqrt{3}}{3} & \frac{\sqrt{2}}{2} & -\frac{\sqrt{6}}{6} \end{pmatrix} \begin{pmatrix} \sqrt{6} & 0 \\ 0 & 2 \\ 0 & 0 \end{pmatrix} \begin{pmatrix} \frac{\sqrt{2}}{2} & \frac{\sqrt{2}}{2} \\ \frac{\sqrt{2}}{2} & -\frac{\sqrt{2}}{2} \end{pmatrix} \equiv U\Sigma V^T.$$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 13.19 — SVD rozklad symetrických matic)</span></p>

Buď $A \in \mathbb{R}^{n \times n}$ symetrická a $A = Q\Lambda Q^T$ její spektrální rozklad, kde vlastní čísla na diagonále $\Lambda$ jsou setříděna sestupně v absolutní hodnotě. Je-li navíc matice $A$ positivně definitní, pak spektrální rozklad je zároveň jejím SVD rozkladem $A = U\Sigma V^T$, protože lze volit $U = Q$, $\Sigma = \Lambda$ a $V = Q$. Singulární čísla a vlastní čísla matice $A$ pak splývají. Pokud matice $A$ není positivně definitní, pak její singulární čísla jsou absolutní hodnoty z vlastních čísel. SVD rozklad může být tvaru $A = U\Sigma V^T$, kde $U = Q'$, $\Sigma = |\Lambda|$, $V = Q$ a matice $Q'$ vznikne z $Q$ přenásobením $-1$ těch sloupců, které odpovídají záporným vlastním číslům.

</div>

Podobně jako u QR rozkladu, tak i pro SVD rozklad existuje redukovaná verze, tzv. *redukovaný* (nebo též *tenký*) SVD rozklad. Buď $A = U\Sigma V^T$ hodnosti $r > 0$. Rozložme $U = (U_1 \mid U_2)$, $V = (V_1 \mid V_2)$ na prvních $r$ sloupců a zbytek a dále označme $S \coloneqq \operatorname{diag}(\sigma_1, \ldots, \sigma_r)$. Pak

$$A = U\Sigma V^T = (U_1 \quad U_2) \begin{pmatrix} S & 0 \\ 0 & 0 \end{pmatrix} \begin{pmatrix} V_1^T \\ V_2^T \end{pmatrix} = U_1 S V_1^T.$$

Redukovaný SVD rozklad používá jen část informace z SVD rozkladu, ale tu podstatnou, ze které můžeme plný SVD rozklad zrekonstruovat (doplněním $U_1$, $V_1$ na ortogonální matice). Redukovaný SVD jsme implicitně používali už v důkazu algoritmu 13.17.

### 13.5 Aplikace SVD rozkladu

#### SVD a ortogonalizace

SVD rozklad lze použít k nalezení ortonormální báze (nejen) sloupcového prostoru $\mathcal{S}(A)$. Na rozdíl od dosavadních přístupů nemusíme předpokládat lineární nezávislost sloupců matice $A$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Věta 13.20)</span></p>

Nechť $A = U\Sigma V^T = U_1 SV_1^T$ je plný resp. redukovaný SVD rozklad matice $A \in \mathbb{R}^{m \times n}$. Pak

1. Sloupce $U_1$ tvoří ortonormální bázi prostoru $\mathcal{S}(A)$.
2. Sloupce $V_1$ tvoří ortonormální bázi prostoru $\mathcal{R}(A)$.
3. Sloupce $V_2$ tvoří ortonormální bázi prostoru $\operatorname{Ker}(A)$.

</div>

*Důkaz.*

1. Přenásobením rovnice $A = U_1 SV_1^T$ maticí $V_1$ zprava dostaneme $AV_1 = U_1 S$. Nyní, $\mathcal{S}(A) \ni \mathcal{S}(AV_1) = \mathcal{S}(U_1 S) = \mathcal{S}(U_1)$ díky regularitě matice $S$. Protože $\operatorname{rank}(A) = \operatorname{rank}(U_1)$, máme rovnost $\mathcal{S}(A) = \mathcal{S}(U_1)$.
2. Plyne z předchozího díky $\mathcal{R}(A) = \mathcal{S}(A^T)$ a tomu, že $A^T = V_1 S U_1^T$ je redukovaný SVD rozklad transponované matice.
3. Z předchozího víme, že sloupce $V_1$ tvoří ortonormální bázi prostoru $\mathcal{R}(A) = \operatorname{Ker}(A)^\perp$. Proto sloupce $V_2$, které doplňují sloupce $V_1$ na ortonormální bázi $\mathbb{R}^n$, představují ortonormální bázi $\operatorname{Ker}(A)$.

#### SVD a projekce do podprostoru

Pomocí SVD rozkladu můžeme snadno vyjádřit matici projekce do sloupcového (a řádkového) prostoru dané matice. Dokonce k tomu nepotřebujeme předpoklad na lineární nezávislost sloupců matice, což bylo potřeba ve větě 8.49.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Věta 13.21)</span></p>

Nechť $A = U_1 SV_1^T$ je redukovaný SVD rozklad matice $A \in \mathbb{R}^{m \times n}$. Pak matice projekce do

1. sloupcového prostoru $\mathcal{S}(A)$ je $U_1 U_1^T$,
2. řádkového prostoru $\mathcal{R}(A)$ je $V_1 V_1^T$.

</div>

*Důkaz.*

1. Z věty 13.20 je $\mathcal{S}(A) = \mathcal{S}(U_1)$. Sloupce $U_1$ tvoří ortonormální systém, a proto matice projekce má dle poznámky 8.52 tvar $U_1 U_1^T$.
2. Plyne z předchozího díky $\mathcal{R}(A) = \mathcal{S}(A^T)$.

#### SVD a geometrie lineárního zobrazení

Buď $A \in \mathbb{R}^{n \times n}$ regulární matice a studujme obraz jednotkové koule při zobrazení $x \mapsto Ax$. Z SVD rozkladu $A = U\Sigma V^T$ plyne, že lineární zobrazení lze rozložit na složení tří základních zobrazení: ortogonální zobrazení s maticí $V^T$, škálování podle $\Sigma$ a ortogonální zobrazení s maticí $U$. Konkrétně, zobrazení s maticí $V^T$ zobrazí kouli na sebe sama, $\Sigma$ ji zdeformuje na elipsoid a $U$ ji otočí/převrátí. Tedy výsledkem bude elipsoid se středem v počátku, poloosy jsou ve směrech sloupců $U$ a délky mají velikost $\sigma_1, \ldots, \sigma_n$.

Hodnota $\frac{\sigma_1}{\sigma_n} \ge 1$ se nazývá *míra deformace* a kvantitativně udává, jak moc zobrazení deformuje geometrické útvary. Je-li hodnota rovna 1, elipsoid bude mít tvar koule, a naopak čím větší bude hodnota, tím protáhlejší bude elipsoid. Význam této hodnoty je ale nejenom geometrický. V numerické matematice se podíl $\frac{\sigma_1}{\sigma_n}$ nazývá *číslo podmíněnosti* a čím je větší, tím hůře podmíněná je matice $A$ ve smyslu, že vykazuje špatné numerické vlastnosti — zaokrouhlování v počítačové aritmetice s pohyblivou řádkovou čárkou způsobuje chyby.

Empirické pravidlo říká, že je-li číslo podmíněnosti řádově $10^k$, pak při výpočtech s maticí (inverze, řešení soustav, atp.) ztrácíme přesnost o $k$ desetinných míst. Ortogonální matice mají číslo podmíněnosti rovné 1, a proto se v numerické matematice často používají. Naproti tomu např. Hilbertovy matice z příkladu 3.51 mají číslo podmíněnosti velmi vysoké:

| $n$ | číslo podmíněnosti $H_n$ |
|-----|--------------------------|
| 3   | $\approx 500$ |
| 5   | $\approx 10^5$ |
| 10  | $\approx 10^{13}$ |
| 15  | $\approx 10^{17}$ |

#### SVD a numerický rank

Hodnost matice $A$ je rovna počtu (kladných) singulárních čísel. Nicméně, pro výpočetní účely se hodně malé kladné číslo považuje za praktickou nulu. Buď $\varepsilon > 0$, pak *numerický rank* matice $A$ je $\max\lbrace s; \; \sigma_s > \varepsilon \rbrace$, tedy počet singulárních čísel větších než $\varepsilon$, ostatní se berou za nulová. Například Matlab / Octave definuje $\varepsilon \coloneqq \max\lbrace m, n \rbrace \cdot \sigma_1 \cdot eps$, kde $eps \approx 2 \cdot 10^{-16}$ je přesnost počítačové aritmetiky.

#### SVD a low-rank aproximace

Buď $A \in \mathbb{R}^{m \times n}$ a $A = U\Sigma V^T$ její SVD rozklad. Jestliže ponecháme $k$ největších singulárních čísel a ostatní vynulujeme $\sigma_{k+1} \coloneqq 0, \ldots, \sigma_r \coloneqq 0$, tak dostaneme matici

$$A' = U \operatorname{diag}(\sigma_1, \ldots, \sigma_k, 0, \ldots, 0) V^T$$

hodnosti $k$, která dobře aproximuje $A$. Navíc tato aproximace je v jistém smyslu nejlepší možná. To jest, v určité normě (viz sekce 13.7) je ze všech matic hodnosti $k$ právě $A'$ nejblíže matici $A$.

#### SVD a komprese dat

Low-rank aproximaci z předchozího odstavce použijeme k jednoduché metodě na ztrátovou kompresi dat. Předpokládejme, že matice $A \in \mathbb{R}^{m \times n}$ reprezentuje data, které chceme zkomprimovat. Pokud $\operatorname{rank}(A) = r$, tak pro redukovaný SVD rozklad $A = U_1 SV_1^T$ si potřebujeme zapamatovat $mr + r + nr = (m + n + 1)r$ hodnot. Při low-rank aproximaci $A \approx U \operatorname{diag}(\sigma_1, \ldots, \sigma_k, 0, \ldots, 0)V^T$ si stačí pamatovat jen $(m + n + 1)k$ hodnot. Tedy kompresní poměr je $k : r$. Čím menší $k$, tím menší objem dat si stačí pamatovat. Ale na druhou stranu, menší $k$ značí horší aproximaci.

#### SVD a míra regularity

Jak jsme uvedli v poznámce 9.10, determinant se jako míra regularity moc nehodí. Zato singulární čísla jsou pro to jako stvořená. Buď $A \in \mathbb{R}^{n \times n}$. Pak $\sigma_n$ udává vzdálenost (v jisté normě, viz sekce 13.7) k nejbližší singulární matici. Takže je to v souladu s tím, co bychom si pod takovou mírou představovali. Ortogonální matice mají míru 1, naproti tomu Hilbertovy matice mají malou míru regularity, tj. jsou téměř singulární:

| $n$ | $\sigma_n(H_n)$ |
|-----|------------------|
| 3   | $\approx 0.0027$ |
| 5   | $\approx 10^{-6}$ |
| 10  | $\approx 10^{-13}$ |
| 15  | $\approx 10^{-18}$ |

### 13.6 Pseudoinverzní matice

Je přirozenou snahou zobecnit pojem inverze matice i na singulární nebo obdélníkové matice. Takové zobecněné inverzi se říká *pseudoinverze* a existuje několik druhů. Nejčastější je tzv. Mooreova–Penroseova pseudoinverze, která spočívá na SVD rozkladu.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Definice 13.23 — Mooreova–Penroseova pseudoinverze)</span></p>

Buď $A \in \mathbb{R}^{m \times n}$ matice s redukovaným SVD rozkladem $A = U_1 SV_1^T$. Je-li $A \neq 0$, pak její *pseudoinverze* je $A^\dagger = V_1 S^{-1} U_1^T \in \mathbb{R}^{n \times m}$. Pro $A = 0$ definujeme pseudoinverzi předpisem $A^\dagger = A^T$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 13.24)</span></p>

Pseudoinverze nenulového vektoru $a \in \mathbb{R}^n$ je $a^\dagger = \frac{1}{a^T a} a^T$, speciálně např. $((1, 1, 1, 1)^T)^\dagger = \frac{1}{4}(1, 1, 1, 1)$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Tvrzení 13.25 — Vlastnosti pseudoinverze)</span></p>

Pro matici $A \in \mathbb{R}^{m \times n}$ platí:

1. Je-li $A$ regulární, tak $A^{-1} = A^\dagger$,
2. $(A^\dagger)^\dagger = A$,
3. $(A^T)^\dagger = (A^\dagger)^T$,
4. $A = AA^\dagger A$,
5. $A^\dagger = A^\dagger A A^\dagger$,
6. $AA^\dagger$ je symetrická,
7. $A^\dagger A$ je symetrická,
8. $A^\dagger = (A^T A)^\dagger A^T$,
9. má-li $A$ lineárně nezávislé sloupce, pak $A^\dagger = (A^T A)^{-1} A^T$,
10. má-li $A$ lineárně nezávislé řádky, pak $A^\dagger = A^T (AA^T)^{-1}$.

</div>

*Důkaz.* Vlastnosti se dokážou jednoduše z definice. Pro ilustraci ukážeme jen dvě vlastnosti, zbytek necháváme čtenáři.

(4) Z definice $AA^\dagger A = U_1 SV_1^T V_1 S^{-1} U_1^T U_1 SV_1^T = U_1 SS^{-1}SV_1^T = U_1 SV_1^T = A$.

(9) Z předpokladu je $V_1$ čtvercová, tedy ortogonální. Pak

$$(A^T A)^{-1} = (V_1 SU_1^T U_1 SV_1^T)^{-1} = (V_1 S^2 V_1^T)^{-1} = V_1 S^{-2} V_1^T,$$

z čehož $(A^T A)^{-1}A^T = V_1 S^{-2} V_1^T V_1 S U_1^T = V_1 S^{-1} U_1^T = A^\dagger$.

První vlastnost říká, že se skutečně jedná o zobecnění klasické inverze. Vlastnosti (4), (7) jsou také zajímavé v tom, že dávají alternativní definici pseudoinverze; ta se totiž ekvivalentně dá definovat jako matice, která splňuje podmínky (4)–(7), a taková matice kupodivu existuje vždy právě jedna.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Věta 13.26)</span></p>

Buď $A \in \mathbb{R}^{m \times n}$. Pak matice projekce do

1. sloupcového prostoru $\mathcal{S}(A)$ je $AA^\dagger$,
2. řádkového prostoru $\mathcal{R}(A)$ je $A^\dagger A$,
3. jádra $\operatorname{Ker}(A)$ je $I_n - A^\dagger A$.

</div>

*Důkaz.*

1. S použitím redukovaného SVD rozkladu $A = U_1 SV_1^T$ upravme $AA^\dagger = U_1 SV_1^T V_1 S^{-1}U_1^T = U_1 U_1^T$. Podle věty 13.21 hledaná matice projekce.
2. Analogicky jako v předchozím je $A^\dagger A = V_1 V_1^T$, což je matice projekce do $\mathcal{R}(A)$.
3. Plyne z věty 8.54 a vlastnosti $\operatorname{Ker}(A) = \mathcal{R}(A)^\perp$ (věta 8.45).

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Věta 13.27 — Pseudoinverzní matice a lineární zobrazení)</span></p>

Uvažujme lineární zobrazení $f(x) = Ax$, kde $A \in \mathbb{R}^{m \times n}$.

1. Pokud definiční obor $f(x)$ omezíme pouze na prostor $\mathcal{R}(A)$, tak dostaneme isomorfismus mezi $\mathcal{R}(A)$ a $f(\mathbb{R}^n)$.
2. Inverzní zobrazení k tomuto isomorfismu má tvar $y \mapsto A^\dagger y$.

</div>

*Důkaz.*

1. Tato část byla dokázána ve tvrzení 8.48, ale předvedeme jiný důkaz. Zobrazení s omezeným definičním oborem je „na", protože podle důsledku 8.47(2) je $f(\mathbb{R}^n) = \mathcal{S}(A) = \mathcal{R}(A^T) = \mathcal{R}(AA^T) = \lbrace Ay; \; y \in \mathcal{R}(A) \rbrace = f(\mathcal{R}(A))$. Jelikož prostory $f(\mathbb{R}^n)$ a $\mathcal{R}(A)$ mají stejnou dimenzi (věta 5.68), musí být zobrazení isomorfismem.
2. Podle věty 13.20(2) se každý vektor $x \in \mathcal{R}(A)$ při zobrazení $x \mapsto A^\dagger Ax$ zobrazí na $A^\dagger Ax = x$. Tudíž $A^\dagger$ je matice inverzního zobrazení k zobrazení $x \mapsto Ax$.

Nejvýznačnější vlastnost pseudoinverze spočívá v popisu množiny řešení řešitelných soustav a množiny přibližných řešení metodou nejmenších čtverců neřešitelných soustav. V obou případech je $A^\dagger b$ v jistém smyslu význačné řešení.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Věta 13.28 — Pseudoinverzní matice a řešení soustav rovnic)</span></p>

Buď $A \in \mathbb{R}^{m \times n}$, $b \in \mathbb{R}^m$ a $X$ množina řešení soustavy $Ax = b$. Je-li $X \neq \emptyset$, pak

$$X = A^\dagger b + \operatorname{Ker}(A).$$

kde $\operatorname{Ker}(A) = \mathcal{S}(I_n - A^\dagger A)$.

Navíc, ze všech vektorů z množiny $X$ má $A^\dagger b$ nejmenší eukleidovskou normu, a je to jediné řešení s touto vlastností.

</div>

*Důkaz.* „$=$" Buď $x \in X$, tj. $Ax = b$. Potom podle tvrzení 13.25(4) je $AA^\dagger b = AA^\dagger Ax = Ax = b$, tedy $A^\dagger b \in X$. Podle věty 7.6 je $X = x_0 + \operatorname{Ker}(A)$, kde $x_0$ je libovolné řešení. Podle věty 13.20(3) je $\operatorname{Ker}(A) = \mathcal{S}(I_n - A^\dagger A)$ a za $x_0$ můžeme volit $A^\dagger b$.

„Norma." Buď $x \in X$. Podle věty 13.27(2) je $A^\dagger b = A^\dagger Ax \in \mathcal{R}(A)$ a dále platí $\mathcal{R}(A) = \operatorname{Ker}(A)^\perp$. Nyní podle Pythagorovy věty pro každé $y \in \mathcal{S}(I_n - A^\dagger A)$ platí

$$\|A^\dagger b + y\|_2^2 = \|A^\dagger b\|_2^2 + \|y\|_2^2 \ge \|A^\dagger b\|_2^2.$$

Tedy $A^\dagger b$ má nejmenší eukleidovskou normu. Každý jiný vektor z $X$ má normu větší, protože $y \neq 0$ implikuje $\|y\|_2 > 0$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Věta 13.29 — Pseudoinverzní matice a metoda nejmenších čtverců)</span></p>

Buď $A \in \mathbb{R}^{m \times n}$, $b \in \mathbb{R}^m$ a $X$ množina přibližných řešení soustavy $Ax = b$ metodou nejmenších čtverců. Pak

$$X = A^\dagger b + \operatorname{Ker}(A).$$

Navíc, ze všech vektorů z množiny $X$ má $A^\dagger b$ nejmenší eukleidovskou normu, a je to jediné řešení s touto vlastností.

</div>

*Důkaz.* Množina přibližných řešení soustavy $Ax = b$ metodou nejmenších čtverců je popsána soustavou $A^T Ax = A^T b$ a je neprázdná, viz věta 8.59. Podle věty 13.28 máme

$$X = (A^T A)^\dagger (A^T b) + \operatorname{Ker}(A^T A).$$

Jelikož podle tvrzení 13.25(8) je $(A^T A)^\dagger A^T = A^\dagger$ a podle důsledku 8.47 je $\operatorname{Ker}(A^T A) = \operatorname{Ker}(A)$, množina $X$ má požadovaný popis a požadovanou vlastnost.

Předchozí dvě věty tedy mj. říkají, že $A^\dagger b$ je význačný vektor. V případě, že soustava $Ax = b$ má řešení, pak je jejím řešením s minimální normou. A v případě, že soustava $Ax = b$ nemá řešení, pak je jejím přibližným řešením (opět s minimální normou) metodou nejmenších čtverců. Navíc není zapotřebí předpokladu na lineární nezávislost sloupců matice $A$.

### 13.7 Maticová norma

Nyní se vrátíme zpět k normám a podíváme se na to, jak zavést normu pro matice. Přestože normy jsme probírali v sekci 8.1, důležitou maticovou normu představuje největší singulární číslo matice, proto zařazujeme tuto část k SVD rozkladu.

V zásadě, matice z $\mathbb{R}^{m \times n}$ tvoří vektorový prostor, proto na matice můžeme pohlížet jako na vektory. Nicméně, pro maticovou normu se uvažuje ještě jedna vlastnost navíc, proto máme speciální definici.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Definice 13.30 — Norma matice)</span></p>

Třída zobrazení $\|\cdot\| \colon \mathbb{R}^{m \times n} \to \mathbb{R}$ je reálná *maticová norma*, pokud to norma pro libovolné $m, n$ a navíc splňuje:

$$\|AB\| \le \|A\| \cdot \|B\| \quad \text{pro všechna } A \in \mathbb{R}^{m \times p}, \; B \in \mathbb{R}^{p \times n}.$$

</div>

Prvním příkladem maticové normy je *Frobeniova norma*:

$$\|A\|_F \coloneqq \sqrt{\sum_{i=1}^{m} \sum_{j=1}^{n} a_{ij}^2}.$$

Je to vlastně eukleidovská norma vektoru tvořeného všemi prvky matice $A$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Tvrzení 13.31 — Frobeniova norma)</span></p>

Buď $A \in \mathbb{R}^{m \times n}$ se singulárními čísly $\sigma_1, \ldots, \sigma_r$. Pak $\|A\|_F = \sqrt{\sum_{i=1}^{r} \sigma_i^2}$.

</div>

*Důkaz.* Podle tvrzení 10.11 a věty 13.15 máme $\|A\|_F = \sqrt{\sum_{i=1}^{m}\sum_{j=1}^{n} a_{ij}^2} = \sqrt{\operatorname{trace}(A^T A)} = \sqrt{\sum_{i=1}^{r}\sigma_i^2}$.

Druhým příkladem maticové normy je *maticová $p$-norma*:

$$\|A\|_p = \max_{x:\|x\|_p=1} \|Ax\|_p.$$

V této definici používáme vektorovou $p$-normu. Výslednou normu si můžeme představit takto: Zobrazíme jednotkovou kouli (v $p$-normě, tedy vektory splňující $\|x\|_p = 1$) lineárním zobrazením $x \mapsto Ax$ a v obrazu vybereme vektor s největší normou. Pro různé hodnoty $p$ dostáváme různé maticové normy.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Tvrzení 13.32 — Maticové $p$-normy)</span></p>

Buď $A \in \mathbb{R}^{m \times n}$. Pak maticové $p$-normy pro $p \in \lbrace 1, 2, \infty \rbrace$ mají tvar

1. $\|A\|_2 = \sigma_1(A)$ (největší singulární číslo),
2. $\|A\|_1 = \max_{j=1,\ldots,n} \sum_{i=1}^{m} |a_{ij}|$,
3. $\|A\|_\infty = \max_{i=1,\ldots,m} \sum_{j=1}^{n} |a_{ij}| = \|A^T\|_1$.

</div>

*Důkaz.*

1. Jak jsme již zmínili, $\|A\|_2$ je velikost největšího bodu elipsy, vzniklé obrazem jednotkové koule při zobrazení $x \mapsto Ax$. Ze sekce 13.5 (SVD a geometrie lineárního zobrazení) víme, že tato hodnota je $\sigma_1(A)$.
2. Označme $c \coloneqq \max_{j=1,\ldots,n} \sum_{i=1}^m |a_{ij}|$. Pro jakékoli $x$ takové, že $\|x\|_1 = 1$, platí $\|Ax\|_1 = \sum_{i=1}^m \left|\sum_{j=1}^n a_{ij}x_j\right| \le \sum_{i=1}^m \sum_{j=1}^n |a_{ij}||x_j| = \sum_{j=1}^n |x_j|\left(\sum_{i=1}^m |a_{ij}|\right) \le \sum_{j=1}^n |x_j| c = c$. Zároveň se nabyde rovnost $\|Ax\|_1 = c$ vhodnou volbou jednotkového vektoru $x = e_i$.
3. Označme $c \coloneqq \max_{i=1,\ldots,m} \sum_{j=1}^n |a_{ij}|$. Pro jakékoli $x$ takové, že $\|x\|_\infty = 1$, platí $\|Ax\|_\infty = \max_{i=1,\ldots,m} \left|\sum_{j=1}^n a_{ij}x_j\right| \le \max_{i=1,\ldots,m} \sum_{j=1}^n |a_{ij}||x_j| \le \max_{i=1,\ldots,m} \sum_{j=1}^n |a_{ij}| = c$. Zároveň se nabyde rovnost $\|Ax\|_\infty = c$ vhodnou volbou vektoru $x \in \lbrace \pm 1 \rbrace^n$.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Poznámka 13.33)</span></p>

Mějme vektor $v \in \mathbb{R}^n$ a libovolné $p$. Nyní výraz $\|v\|_p$ může označovat jak vektorovou, tak maticovou $p$-normu, považujeme-li $v$ za matici s jedním sloupcem. To však nevadí, protože obě dávají stejnou hodnotu:

$$\|v\|_p = \max_{x \in \mathbb{R}:\|x\|=1} \|vx\|_p = \max_{x \in \lbrace \pm 1 \rbrace} \|\pm v\|_p = \|v\|_p.$$

</div>

Víme z věty 8.66, že přenásobení vektoru ortogonální maticí nemění jeho eukleidovskou normu. Nyní tvrzení zobecníme pro Frobeniovu a maticovou 2-normu.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Tvrzení 13.34)</span></p>

Buď $A \in \mathbb{R}^{m \times n}$ a buďte $Q \in \mathbb{R}^{m \times m}$, $R \in \mathbb{R}^{n \times n}$ ortogonální. Pak

1. $\|QAR\|_F = \|A\|_F$,
2. $\|QAR\|_2 = \|A\|_2$.

</div>

*Důkaz.*

1. Analogicky jako v důkazu tvrzení 13.31 máme $\|QAR\|_F^2 = \operatorname{trace}((QAR)^T(QAR)) = \operatorname{trace}(R^T A^T Q^T QAR) = \operatorname{trace}(R^T A^T AR) = \operatorname{trace}(A^T ARR^T) = \operatorname{trace}(A^T A) = \|A\|_F^2$, kde jsme navíc využili fakt, že $\operatorname{trace}(BC) = \operatorname{trace}(CB)$ pro každé $B, C \in \mathbb{R}^{n \times n}$.
2. S použitím substituce $x \coloneqq R^T y$ odvodíme $\|QAR\|_2 = \max_{x:\|x\|_2=1} \|QARx\|_2 = \max_{x:\|x\|_2=1} \|ARx\|_2 = \max_{y:\|R^T y\|_2=1} \|Ay\|_2 = \max_{y:\|y\|_2=1} \|Ay\|_2 = \|A\|_2$.

Maticové normy se objevují v různých souvislostech. Nejprve ukážeme, že dávají odhad na velikost vlastních čísel.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Věta 13.35 — Odhad spektrálního poloměru pomocí normy)</span></p>

Buď $A \in \mathbb{R}^{n \times n}$. Pak pro každou maticovou normu platí $\rho(A) \le \|A\|$.

</div>

*Důkaz.* Buď $\lambda \in \mathbb{C}$ libovolné vlastní číslo a $x$ odpovídající vlastní vektor matice $A$, tedy $Ax = \lambda x$. Definujme matici $X \coloneqq (x \mid o \mid \cdots \mid o)$. Protože platí $AX = \lambda X$, můžeme odvodit

$$|\lambda| \cdot \|X\| = \|\lambda X\| = \|AX\| \le \|A\| \cdot \|X\|.$$

Vydělením $\|X\| \neq 0$ dostáváme $|\lambda| \le \|A\|$.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 13.36)</span></p>

Uvažujme matici

$$A = \begin{pmatrix} 1 & 2 & 3 \\ 1 & 2 & 3 \\ 3 & 6 & 9 \end{pmatrix}.$$

Její spektrální poloměr a různé typy norem mají hodnoty:

$$\rho(A) = 12, \quad \|A\|_F = \sqrt{154} \approx 12.4097, \quad \|A\|_2 = \sqrt{154} \approx 12.4097, \quad \|A\|_1 = 15, \quad \|A\|_\infty = 18.$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Poznámka 13.37 — Výpočetní složitost)</span></p>

Buď $A \in \mathbb{R}^{n \times n}$. Není těžké nahlédnout, že výpočet Frobeniovy normy a $p$-normy pro $p \in \lbrace 1, \infty \rbrace$ má asymptotickou složitost $2n^2$. Maticová 2-norma se počítá pouze iterativně a běžně používané metody mají kubickou složitost (s určitým koeficientem), podobně jako vlastní čísla matice (viz začátek sekce 10.7). Přesto je defaultní maticovou normou, kterou např. Matlab či Octave používají.

</div>

Další velmi zajímavá vlastnost singulárních čísel je, že $\sigma_i$ udává v 2-normě vzdálenost matice k nejbližší matici hodnosti nanejvýš $i - 1$. V důkazu následující věty je schované i to, jak tuto matici sestrojit — ne náhodou je to matice z low-rank aproximace (srov. sekce 13.5).

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Věta 13.38 — Interpretace singulárních čísel)</span></p>

Buď $A \in \mathbb{R}^{m \times n}$ se singulárními čísly $\sigma_1, \ldots, \sigma_r$. Pak

$$\sigma_i = \min \lbrace \|A - B\|_2; \; B \in \mathbb{R}^{m \times n}, \; \operatorname{rank}(B) \le i - 1 \rbrace$$

pro každé $i = 1, \ldots, r$.

</div>

*Důkaz.* Nerovnost „$\ge$". Nechť $A = U\Sigma V^T$ je SVD rozklad matice $A$. Definujme matici $B \coloneqq U \operatorname{diag}(\sigma_1, \ldots, \sigma_{i-1}, 0, \ldots, 0) V^T$. Pak

$$\|A - B\|_2 = \|U \operatorname{diag}(0, \ldots, 0, \sigma_i, \ldots, \sigma_n) V^T\|_2 = \|\operatorname{diag}(0, \ldots, 0, \sigma_i, \ldots, \sigma_n)\|_2 = \sigma_i.$$

Nerovnost „$\le$". Buď $B \in \mathbb{R}^{n \times n}$ libovolná matice hodnosti nanejvýš $i - 1$ a ukážeme, že $\|A - B\|_2 \ge \sigma_i$. Nechť $V_1$ sestává z prvních $i$ sloupců matice $V$. Buď $o \neq z \in \operatorname{Ker}(B) \cap \mathcal{S}(V_1)$, to jest $Bz = o$, a navíc normujeme $z$ tak, aby $\|z\|_2 = 1$. Takový vektor existuje, protože $\dim\operatorname{Ker}(B) \ge n - i + 1$ a $\dim\mathcal{S}(V_1) = i$. Pak

$$\|A - B\|_2^2 = \max_{x:\|x\|_2=1} \|(A - B)x\|_2^2 \ge \|(A - B)z\|_2^2 = \|Az\|_2^2 = \|U\Sigma V^T z\|_2^2.$$

Protože $z \in \mathcal{S}(V_1)$, lze psát $z = Vy$ pro nějaký vektor $y = (y_1, \ldots, y_i, 0, \ldots, 0)^T$, přičemž $\|y\|_2 = \|V^T z\|_2 = \|z\|_2 = 1$. Nyní

$$\|U\Sigma V^T z\|_2^2 = \|U\Sigma V^T Vy\|_2^2 = \|\Sigma y\|_2^2 = \sum_{j=1}^{i} \sigma_j^2 y_j^2 \ge \sum_{j=1}^{i} \sigma_i^2 y_j^2 = \sigma_i^2 \|y\|_2^2 = \sigma_i^2.$$

Speciálně, nejmenší singulární číslo $\sigma_n$ matice $A \in \mathbb{R}^{n \times n}$ udává vzdálenost k nejbližší singulární matici. To znamená, že matice $A + C$ je regulární pro všechny matice $C \in \mathbb{R}^{n \times n}$ splňující $\|C\|_2 < \sigma_n$.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Příklad 13.39)</span></p>

Uvažujme matici

$$A = \begin{pmatrix} 1 & 1 & 1 & 1 \\ 1 & 1 & -1 & -1 \\ 1 & -1 & 1 & -1 \\ 1 & -1 & -1 & 1 \end{pmatrix}.$$

Její nejmenší singulární číslo je $\sigma_4 = 2$ (ve skutečnosti jsou všechna singulární čísla stejně velká, protože $A$ je dvojnásobkem ortogonální matice). Tudíž matice zůstane regulární i když k ní přičteme libovolnou matici s 2-normou menší než 2.

</div>

### Shrnutí ke kapitole 13

Maticové rozklady jsou velmi účinný nástroj teoretické i výpočetní informatiky. Rozkladů existuje celá řada, mezi ty význačné se řadí QR a SVD rozklad. Ne náhodou oba používají v rozkladu ortogonální matice.

QR rozklad vyjadřuje libovolnou reálnou matici jako součin ortogonální a horní trojúhelníkové. Tento rozklad lze výpočetně jednoduše získat pomocí Gramovy–Schmidtovy ortogonalizace či Householderovy transformací. Využití QR rozkladu je nepřeberné: řešení soustav lineárních rovnic, nalezení ortonormální báze, sestrojení matice ortogonální projekce, řešení metodou nejmenších čtverců, metoda na výpočet vlastních čísel, ...

SVD rozklad má analogické vlastnosti. Danou reálnou matici rozkládá na součin ortogonální, diagonální (ale ne nutně čtvercovou!) a ortogonální. Využití je podobné jako pro QR rozklad, ale navíc nám říká něco o geometrii lineárních zobrazení, dává nástroj pro aproximaci a kompresi dat. Umožňuje také přirozeně rozšířit pojem inverzní matice na ne nutně regulární matice.

Singulární čísla (= čísla diagonální matice z SVD rozkladu) pak poskytují podstatné informace o lineárním zobrazení $x \mapsto Ax$, o matici $A$ samotné a také o datech, která reprezentuje. Singulární čísla říkají, jak moc lineární zobrazení degeneruje objekty, jaká je vzdálenost k nejbližší singulární matici a jaké má matice numerické vlastnosti, a největší singulární číslo reprezentuje často používanou maticovou normu.
