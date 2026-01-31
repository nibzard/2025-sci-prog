'use client';

import { useState, useEffect, useMemo } from 'react';
import dynamic from 'next/dynamic';
import {
  Home,
  Building2,
  MapPin,
  TrendingUp,
  BarChart3,
  Calculator,
  Info,
  Calendar,
  Ruler,
  DoorOpen,
  Euro
} from 'lucide-react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  AreaChart,
  Area,
} from 'recharts';
import type { EDAData, FormData, CityData, PredictionInput, PredictionResult } from '@/lib/types';
import { formatPrice, formatNumber } from '@/lib/utils';
import { predictPrice, checkHealth } from '@/lib/api';

// Dynamic import for map (no SSR)
const MapComponent = dynamic(() => import('@/components/Map'), {
  ssr: false,
  loading: () => (
    <div className="h-[400px] bg-[var(--card)] rounded-xl animate-pulse flex items-center justify-center border border-[var(--border)]">
      <span className="text-[var(--muted-foreground)]">Učitavanje karte...</span>
    </div>
  )
});

const CHART_COLORS = {
  primary: '#3b82f6',
  secondary: '#8b5cf6',
  tertiary: '#06b6d4',
  success: '#22c55e',
  warning: '#f59e0b',
  danger: '#ef4444',
};

// The 10 main cities (9 cities + Grad Zagreb)
const MAIN_CITIES = [
  'Zagreb', 'Split', 'Rijeka', 'Osijek', 'Zadar', 'Pula',
  'Slavonski Brod', 'Karlovac', 'Sisak', 'Velika Gorica'
];

// Custom tooltip style
const tooltipStyle = {
  contentStyle: {
    background: 'var(--card)',
    border: '1px solid var(--border)',
    borderRadius: '8px'
  },
  labelStyle: { color: 'var(--foreground)' },
  cursor: { fill: 'rgba(59, 130, 246, 0.1)' }
};

export default function HomePage() {
  const [edaData, setEdaData] = useState<EDAData | null>(null);
  const [formData, setFormData] = useState<FormData | null>(null);
  const [loading, setLoading] = useState(true);

  // Form state
  const [propertyType, setPropertyType] = useState<'apartments' | 'houses'>('apartments');
  const [selectedCity, setSelectedCity] = useState<string | null>(null);
  const [formValues, setFormValues] = useState({
    // Location
    zupanija: '',
    grad_opcina: '',
    naselje: '',
    // Core
    stambena_povrsina: '',
    broj_soba: '',
    godina_izgradnje: '',
    godina_renovacije: '',
    energetski_razred: '',
    wc_broj: '',
    kupaonica_s_wc_broj: '',
    broj_etaza: '',
    broj_parkirnih_mjesta: '',
    // Apartments specific
    kat: '',
    ukupni_broj_katova: '',
    tip_stana: '',
    orijentacija_istok: false,
    orijentacija_jug: false,
    orijentacija_sjever: false,
    orijentacija_zapad: false,
    // Houses specific
    povrsina_okucnice: '',
    pogled_more: false,
    tip_kuce: '',
    vrsta_gradnje: '',
    // Binary flags
    novogradnja: false,
    ima_lift: false,
    balkon_balkon: false,
    balkon_terasa: false,
    balkon_lodja: false,
    grijanje_klima: false,
    objekt_bazen: false,
    objekt_dvoriste: false,
    objekt_podrum: false,
    objekt_rostilj: false,
    objekt_spremiste: false,
    objekt_vrtna_kucica: false,
    objekt_zimski_vrt: false,
    dozvola_vlasnicki_list: false,
    dozvola_gradevinska: false,
    dozvola_uporabna: false,
    // Funk features
    funk_kamin: false,
    funk_podno_grijanje: false,
    funk_alarm: false,
    funk_sauna: false,
    // Alt energija
    alt_solarni_paneli: false,
    alt_toplinske_pumpe: false,
    // Select fields
    parking_type: '',
    grijanje_sustav: '',
  });
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [predictionError, setPredictionError] = useState<string | null>(null);
  const [backendReady, setBackendReady] = useState<boolean | null>(null);
  const [removeOutliers, setRemoveOutliers] = useState(true);

  // Prediction state
  const [prediction, setPrediction] = useState<{
    price: number;
    pricePerM2: number;
    range: { low: number; high: number };
    comparisons: { city: string; price: number }[];
    modelVersion?: string;
    featuresUsed?: number;
  } | null>(null);
  const [predicting, setPredicting] = useState(false);

  // Check backend health on mount
  useEffect(() => {
    checkHealth().then(result => setBackendReady(result.ready));
  }, []);

  // Load data
  useEffect(() => {
    Promise.all([
      fetch('/data/eda.json').then(r => r.json()),
      fetch('/data/form.json').then(r => r.json()),
    ]).then(([eda, form]) => {
      setEdaData(eda);
      setFormData(form);
      setLoading(false);
    }).catch(err => {
      console.error('Error loading data:', err);
      setLoading(false);
    });
  }, []);

  // Get location hierarchy for current property type
  const locationHierarchy = useMemo(() => {
    if (!formData) return null;
    return formData[propertyType]?.location_hierarchy || null;
  }, [formData, propertyType]);

  // Get available županije
  const availableZupanije = useMemo(() => {
    if (!locationHierarchy) return [];
    return Object.keys(locationHierarchy).sort();
  }, [locationHierarchy]);

  // Get available gradovi based on selected županija (filter out Solin)
  const availableGradovi = useMemo(() => {
    if (!locationHierarchy || !formValues.zupanija) return [];
    const grads = locationHierarchy[formValues.zupanija];
    return grads ? Object.keys(grads).filter(g => g !== 'Solin').sort() : [];
  }, [locationHierarchy, formValues.zupanija]);

  // Get available naselja based on selected grad
  const availableNaselja = useMemo(() => {
    if (!locationHierarchy || !formValues.zupanija || !formValues.grad_opcina) return [];
    const naselja = locationHierarchy[formValues.zupanija]?.[formValues.grad_opcina];
    return naselja ? naselja.sort() : [];
  }, [locationHierarchy, formValues.zupanija, formValues.grad_opcina]);

  // Reset dependent fields when parent changes
  const handleZupanijaChange = (value: string) => {
    setFormValues({
      ...formValues,
      zupanija: value,
      grad_opcina: '',
      naselje: ''
    });
  };

  const handleGradChange = (value: string) => {
    setFormValues({
      ...formValues,
      grad_opcina: value,
      naselje: ''
    });
  };

  // Handle prediction via HF Spaces API
  const handlePredict = async () => {
    if (!formData || !formValues.stambena_povrsina || !formValues.zupanija) return;

    setPredicting(true);
    setPredictionError(null);

    const area = parseFloat(formValues.stambena_povrsina);

    try {
      // Build prediction input
      const input: PredictionInput = {
        property_type: propertyType,
        zupanija: formValues.zupanija,
        grad_opcina: formValues.grad_opcina,
        naselje: formValues.naselje || undefined,
        stambena_povrsina: area,
        broj_soba: formValues.broj_soba ? parseFloat(formValues.broj_soba) : undefined,
        godina_izgradnje: formValues.godina_izgradnje ? parseInt(formValues.godina_izgradnje) : undefined,
        godina_renovacije: formValues.godina_renovacije ? parseInt(formValues.godina_renovacije) : undefined,
        energetski_razred: formValues.energetski_razred ? parseInt(formValues.energetski_razred) : undefined,
        wc_broj: formValues.wc_broj ? parseInt(formValues.wc_broj) : undefined,
        kupaonica_s_wc_broj: formValues.kupaonica_s_wc_broj ? parseInt(formValues.kupaonica_s_wc_broj) : undefined,
        broj_etaza: formValues.broj_etaza ? parseFloat(formValues.broj_etaza) : undefined,
        broj_parkirnih_mjesta: formValues.broj_parkirnih_mjesta ? parseInt(formValues.broj_parkirnih_mjesta) : undefined,
        // Apartments specific
        kat: formValues.kat ? parseFloat(formValues.kat) : undefined,
        ukupni_broj_katova: formValues.ukupni_broj_katova ? parseFloat(formValues.ukupni_broj_katova) : undefined,
        tip_stana: formValues.tip_stana || undefined,
        // Orientation (apartments)
        orijentacija_istok: formValues.orijentacija_istok ? 1 : undefined,
        orijentacija_jug: formValues.orijentacija_jug ? 1 : undefined,
        orijentacija_sjever: formValues.orijentacija_sjever ? 1 : undefined,
        orijentacija_zapad: formValues.orijentacija_zapad ? 1 : undefined,
        // Houses specific
        povrsina_okucnice: formValues.povrsina_okucnice ? parseFloat(formValues.povrsina_okucnice) : undefined,
        pogled_more: formValues.pogled_more ? 1 : undefined,
        tip_kuce: formValues.tip_kuce || undefined,
        vrsta_gradnje: formValues.vrsta_gradnje || undefined,
        // Binary flags
        podatak_novogradnja: formValues.novogradnja ? 1 : undefined,
        podatak_lift: formValues.ima_lift ? 1 : undefined,
        balkon_balkon: formValues.balkon_balkon ? 1 : undefined,
        balkon_terasa: formValues.balkon_terasa ? 1 : undefined,
        balkon_lodja_loggia: formValues.balkon_lodja ? 1 : undefined,
        grijanje_klima: formValues.grijanje_klima ? 1 : undefined,
        objekt_bazen: formValues.objekt_bazen ? 1 : undefined,
        objekt_dvoriste_vrt: formValues.objekt_dvoriste ? 1 : undefined,
        objekt_podrum: formValues.objekt_podrum ? 1 : undefined,
        objekt_rostilj: formValues.objekt_rostilj ? 1 : undefined,
        objekt_spremiste: formValues.objekt_spremiste ? 1 : undefined,
        objekt_vrtna_kucica: formValues.objekt_vrtna_kucica ? 1 : undefined,
        objekt_zimski_vrt: formValues.objekt_zimski_vrt ? 1 : undefined,
        // Permits
        dozvola_vlasnicki_list: formValues.dozvola_vlasnicki_list ? 1 : undefined,
        dozvola_gradevinska_dozvola: formValues.dozvola_gradevinska ? 1 : undefined,
        dozvola_uporabna_dozvola: formValues.dozvola_uporabna ? 1 : undefined,
        // Funk features
        funk_kamin: formValues.funk_kamin ? 1 : undefined,
        funk_podno_grijanje: formValues.funk_podno_grijanje ? 1 : undefined,
        funk_alarm: formValues.funk_alarm ? 1 : undefined,
        funk_sauna: formValues.funk_sauna ? 1 : undefined,
        // Alt energija
        alt_solarni_paneli: formValues.alt_solarni_paneli ? 1 : undefined,
        alt_toplinske_pumpe: formValues.alt_toplinske_pumpe ? 1 : undefined,
        // Select fields
        parking_type: formValues.parking_type || undefined,
        grijanje_sustav: formValues.grijanje_sustav || undefined,
      };

      // Build city-to-zupanija mapping from location hierarchy
      const cityToZupanija: Record<string, string> = {};
      if (locationHierarchy) {
        for (const [zup, grads] of Object.entries(locationHierarchy)) {
          for (const grad of Object.keys(grads as Record<string, string[]>)) {
            cityToZupanija[grad] = zup;
          }
        }
      }

      // Determine comparison cities (exclude the user's selected city)
      const comparisonCities = MAIN_CITIES.filter(city => {
        if (city === formValues.grad_opcina) return false;
        // Zagreb special case: skip if user already selected a city in Grad Zagreb
        if (city === 'Zagreb' && formValues.zupanija === 'Grad Zagreb') return false;
        // Zagreb is a županija, not a grad_opcina, so allow it without hierarchy check
        if (city === 'Zagreb') return true;
        return cityToZupanija[city] !== undefined;
      });

      // Fire main prediction + comparison predictions in parallel
      const mainPromise = predictPrice(input);
      const compPromises = comparisonCities.map(city => {
        // Zagreb special case: use Grad Zagreb as županija, omit grad_opcina
        const compInput: PredictionInput = city === 'Zagreb'
          ? { ...input, zupanija: 'Grad Zagreb', grad_opcina: undefined, naselje: undefined }
          : { ...input, zupanija: cityToZupanija[city], grad_opcina: city, naselje: undefined };
        return predictPrice(compInput)
          .then(r => ({ city, price: r.predicted_price }))
          .catch(() => null);
      });

      const [result, ...compResults] = await Promise.all([mainPromise, ...compPromises]);

      const comparisons = (compResults as ({ city: string; price: number } | null)[])
        .filter((c): c is { city: string; price: number } => c !== null)
        .sort((a, b) => b.price - a.price);

      setPrediction({
        price: result.predicted_price,
        pricePerM2: result.price_per_m2,
        range: result.price_range,
        comparisons,
        modelVersion: result.model_version,
        featuresUsed: result.features_used,
      });
    } catch (error) {
      setPredictionError(error instanceof Error ? error.message : 'Greška pri procjeni cijene. Pokušajte ponovno.');
    } finally {
      setPredicting(false);
    }
  };

  // Filter cities (exclude Solin)
  const filteredCities = useMemo(() => {
    return edaData?.cities?.filter(city => city.name !== 'Solin') || [];
  }, [edaData]);

  // Memoized: compute all EDA data (outlier filtering, distributions, stats)
  const currentData = useMemo(() => {
    if (!edaData) return null;
    const data = edaData[propertyType];

    // --- Helper: IQR outlier filter (single-pass where possible) ---
    const filterOutliers = (scatter: typeof data.scatter) => {
      if (scatter.length < 4) return scatter;
      let filtered = scatter.filter(d => d.cijena >= 5000 && d.stambena_povrsina >= 10 && d.stambena_povrsina <= 2000);
      if (filtered.length < 4) return filtered;
      const percentile = (arr: number[], p: number) => arr[Math.floor(arr.length * p)];
      const prices = filtered.map(d => d.cijena).sort((a, b) => a - b);
      const pQ1 = percentile(prices, 0.25), pQ3 = percentile(prices, 0.75), pIqr = pQ3 - pQ1;
      const areas = filtered.map(d => d.stambena_povrsina).sort((a, b) => a - b);
      const aQ1 = percentile(areas, 0.25), aQ3 = percentile(areas, 0.75), aIqr = aQ3 - aQ1;
      const pm2s = filtered.map(d => d.cijena / d.stambena_povrsina).sort((a, b) => a - b);
      const mQ1 = percentile(pm2s, 0.25), mQ3 = percentile(pm2s, 0.75), mIqr = mQ3 - mQ1;
      const pLo = pQ1 - 1.5 * pIqr, pHi = pQ3 + 1.5 * pIqr;
      const aLo = aQ1 - 1.5 * aIqr, aHi = aQ3 + 1.5 * aIqr;
      const mLo = mQ1 - 1.5 * mIqr, mHi = mQ3 + 1.5 * mIqr;
      return filtered.filter(d => {
        const pm2 = d.cijena / d.stambena_povrsina;
        return d.cijena >= pLo && d.cijena <= pHi
          && d.stambena_povrsina >= aLo && d.stambena_povrsina <= aHi
          && pm2 >= mLo && pm2 <= mHi;
      });
    };

    // --- Helper: compute stats + distributions in a single pass ---
    const computeAll = (scatter: typeof data.scatter) => {
      const n = scatter.length;
      if (n === 0) return null;

      // Single pass: accumulate sums + bin counts
      let sumP = 0, sumA = 0, sumM = 0;
      const priceBinCounts: Record<number, number> = {};
      const areaBinCounts: Record<number, number> = {};
      const m2BinCounts: Record<number, number> = {};
      const roomsCounts: Record<number, number> = {};
      const yearCounts: Record<number, number> = {};

      for (let i = 0; i < n; i++) {
        const d = scatter[i];
        const pm2 = d.cijena / d.stambena_povrsina;
        sumP += d.cijena;
        sumA += d.stambena_povrsina;
        sumM += pm2;

        // Price bin (50k steps)
        const pBin = d.cijena >= 1000000 ? 1000000 : Math.floor(d.cijena / 50000) * 50000;
        priceBinCounts[pBin] = (priceBinCounts[pBin] || 0) + 1;

        // Area bin (20 steps)
        const aBin = d.stambena_povrsina >= 300 ? 300 : Math.floor(d.stambena_povrsina / 20) * 20;
        areaBinCounts[aBin] = (areaBinCounts[aBin] || 0) + 1;

        // Price/m2 bin (500 steps)
        const mBin = pm2 >= 10000 ? 10000 : Math.floor(pm2 / 500) * 500;
        m2BinCounts[mBin] = (m2BinCounts[mBin] || 0) + 1;

        // Rooms
        if (d.broj_soba != null) roomsCounts[d.broj_soba] = (roomsCounts[d.broj_soba] || 0) + 1;

        // Year decade
        if (d.godina_izgradnje != null) {
          const dec = Math.floor(d.godina_izgradnje / 10) * 10;
          if (dec >= 1900 && dec < 2030) yearCounts[dec] = (yearCounts[dec] || 0) + 1;
        }
      }

      // Medians (need sorted arrays)
      const prices = scatter.map(d => d.cijena).sort((a, b) => a - b);
      const areas = scatter.map(d => d.stambena_povrsina).sort((a, b) => a - b);
      const pm2arr = scatter.map(d => d.cijena / d.stambena_povrsina).sort((a, b) => a - b);
      const mid = Math.floor(n / 2);
      const medianP = n % 2 ? prices[mid] : (prices[mid - 1] + prices[mid]) / 2;
      const medianA = n % 2 ? areas[mid] : (areas[mid - 1] + areas[mid]) / 2;
      const medianM = n % 2 ? pm2arr[mid] : (pm2arr[mid - 1] + pm2arr[mid]) / 2;

      // Std dev (second pass)
      const meanP = sumP / n;
      let sumSqP = 0;
      for (let i = 0; i < n; i++) sumSqP += (scatter[i].cijena - meanP) ** 2;

      // Build sorted bin arrays
      const toBins = (counts: Record<number, number>, step: number) =>
        Object.entries(counts)
          .map(([k, count]) => ({ min: Number(k), max: Number(k) + step, count }))
          .sort((a, b) => a.min - b.min);

      const yearBins = Object.entries(yearCounts)
        .map(([k, count]) => ({ decade: `${k}s`, min: Number(k), max: Number(k) + 10, count }))
        .sort((a, b) => a.min - b.min);

      return {
        stats: {
          total_count: n,
          avg_price: meanP,
          median_price: medianP,
          min_price: prices[0],
          max_price: prices[n - 1],
          std_price: Math.sqrt(sumSqP / n),
          avg_area: sumA / n,
          median_area: medianA,
          avg_price_per_m2: sumM / n,
          median_price_per_m2: medianM,
        },
        distributions: {
          price_distribution: toBins(priceBinCounts, 50000),
          area_distribution: toBins(areaBinCounts, 20),
          price_per_m2_distribution: toBins(m2BinCounts, 500),
          rooms_distribution: Object.entries(roomsCounts)
            .map(([rooms, count]) => ({ rooms: parseFloat(rooms), count }))
            .sort((a, b) => a.rooms - b.rooms),
          year_distribution: yearBins.length > 0 ? yearBins : undefined,
        },
      };
    };

    // --- Build current data ---
    if (selectedCity) {
      const isZagreb = selectedCity === 'Zagreb';
      const cityGrad = isZagreb
        ? data.by_zupanija?.find(d => d.zupanija === 'Grad Zagreb')
        : data.by_grad?.find(d => d.grad_opcina === selectedCity);

      let filteredScatter = isZagreb
        ? data.scatter?.filter(d => d.zupanija === 'Grad Zagreb') || []
        : data.scatter?.filter(d => d.grad_opcina === selectedCity) || [];

      if (removeOutliers) filteredScatter = filterOutliers(filteredScatter);

      const computed = computeAll(filteredScatter);

      return {
        ...data,
        total_count: computed?.stats.total_count ?? cityGrad?.count ?? data.total_count,
        avg_price: computed?.stats.avg_price ?? cityGrad?.price_mean ?? data.avg_price,
        median_price: computed?.stats.median_price ?? cityGrad?.price_median ?? data.median_price,
        min_price: computed?.stats.min_price ?? cityGrad?.price_min ?? data.min_price,
        max_price: computed?.stats.max_price ?? cityGrad?.price_max ?? data.max_price,
        std_price: computed?.stats.std_price ?? cityGrad?.price_std ?? data.std_price,
        avg_area: computed?.stats.avg_area ?? cityGrad?.area_mean ?? data.avg_area,
        median_area: computed?.stats.median_area ?? cityGrad?.area_median ?? data.median_area,
        avg_price_per_m2: computed?.stats.avg_price_per_m2 ?? cityGrad?.price_per_m2 ?? data.avg_price_per_m2,
        median_price_per_m2: computed?.stats.median_price_per_m2 ?? cityGrad?.price_per_m2 ?? data.median_price_per_m2,
        scatter: filteredScatter,
        ...(computed?.distributions || {}),
        selectedCityData: cityGrad,
      };
    }

    if (removeOutliers) {
      const filtered = filterOutliers(data.scatter || []);
      const computed = computeAll(filtered);

      return {
        ...data,
        ...(computed?.stats || {}),
        scatter: filtered,
        ...(computed?.distributions || {}),
      };
    }

    return data;
  }, [edaData, propertyType, selectedCity, removeOutliers]);

  // Get only main cities for the top lists
  const getMainCitiesData = () => {
    if (!currentData?.by_grad) return { expensive: [], affordable: [] };

    const mainCitiesData = currentData.by_grad.filter(city =>
      MAIN_CITIES.includes(city.grad_opcina || '') && city.grad_opcina !== 'Zagreb'
    );

    // Treat Grad Zagreb (županija) as a single city entry
    const zagrebZup = currentData.by_zupanija?.find(d => d.zupanija === 'Grad Zagreb');
    if (zagrebZup) {
      mainCitiesData.push({ ...zagrebZup, grad_opcina: 'Zagreb' });
    }

    const sorted = [...mainCitiesData].sort((a, b) => (b.price_mean || 0) - (a.price_mean || 0));

    return {
      expensive: sorted.slice(0, 5),
      affordable: sorted.slice(-5).reverse(),
    };
  };

  const mainCitiesData = getMainCitiesData();

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-[var(--background)]">
        <div className="gradient-orb-1" />
        <div className="gradient-orb-2" />
        <div className="gradient-orb-3" />
        <div className="text-center">
          <div className="w-16 h-16 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
          <p className="text-[var(--muted-foreground)]">Učitavanje podataka...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-[var(--background)]">
      {/* Background gradient orbs */}
      <div className="gradient-orb-1" />
      <div className="gradient-orb-2" />
      <div className="gradient-orb-3" />

      {/* Header */}
      <header className="sticky top-0 z-50 bg-[var(--background)]/80 backdrop-blur-lg border-b border-[var(--border)]">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-blue-600 rounded-xl flex items-center justify-center">
                <Home className="w-5 h-5 text-white" />
              </div>
              <div>
                <h1 className="font-bold text-lg">Procjenitelj Cijena</h1>
                <p className="text-xs text-[var(--muted-foreground)]">Nekretnine u Hrvatskoj</p>
              </div>
            </div>
            <nav className="hidden md:flex items-center gap-6">
              <a href="#procjena" className="text-sm font-medium hover:text-blue-500 transition-colors">Procjena</a>
              <a href="#karta" className="text-sm font-medium hover:text-blue-500 transition-colors">Karta</a>
              <a href="#analiza" className="text-sm font-medium hover:text-blue-500 transition-colors">Analiza</a>
            </nav>
          </div>
        </div>
      </header>

      {/* Hero Section with Form */}
      <section id="procjena" className="gradient-hero py-8 lg:py-10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-6">
            <h2 className="text-4xl lg:text-5xl font-bold mb-2">
              Koliko vrijedi vaša nekretnina?
            </h2>
            <p className="text-lg text-[var(--muted-foreground)] max-w-2xl mx-auto">
              Unesite karakteristike nekretnine i dobijte procjenu cijene temeljenu na
              stvarnim podacima s hrvatskog tržišta nekretnina.
            </p>
          </div>

          {/* Property Type Toggle */}
          <div className="flex justify-center mb-4">
            <div className="tabs">
              <button
                className={`tab ${propertyType === 'apartments' ? 'active' : ''}`}
                onClick={() => {
                  setPropertyType('apartments');
                  setFormValues({ ...formValues, zupanija: '', grad_opcina: '', naselje: '' });
                }}
              >
                <Building2 className="w-4 h-4 inline-block mr-2" />
                Stanovi
              </button>
              <button
                className={`tab ${propertyType === 'houses' ? 'active' : ''}`}
                onClick={() => {
                  setPropertyType('houses');
                  setFormValues({ ...formValues, zupanija: '', grad_opcina: '', naselje: '' });
                }}
              >
                <Home className="w-4 h-4 inline-block mr-2" />
                Kuće
              </button>
            </div>
          </div>

          {/* Prediction Form */}
          <div className="max-w-4xl mx-auto">
            <div className="card">
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {/* Zupanija */}
                <div>
                  <label className="label">Županija</label>
                  <select
                    className="select"
                    value={formValues.zupanija}
                    onChange={(e) => handleZupanijaChange(e.target.value)}
                  >
                    <option value="">Odaberite županiju</option>
                    {availableZupanije.map(z => (
                      <option key={z} value={z}>{z}</option>
                    ))}
                  </select>
                </div>

                {/* Grad */}
                <div>
                  <label className="label">Grad / Općina</label>
                  <select
                    className="select"
                    value={formValues.grad_opcina}
                    onChange={(e) => handleGradChange(e.target.value)}
                    disabled={!formValues.zupanija}
                  >
                    <option value="">Odaberite grad</option>
                    {availableGradovi.map(g => (
                      <option key={g} value={g}>{g}</option>
                    ))}
                  </select>
                  {!formValues.zupanija && (
                    <p className="text-xs text-[var(--muted-foreground)] mt-1">Prvo odaberite županiju</p>
                  )}
                </div>

                {/* Naselje */}
                <div>
                  <label className="label">Naselje / Kvart</label>
                  <select
                    className="select"
                    value={formValues.naselje}
                    onChange={(e) => setFormValues({...formValues, naselje: e.target.value})}
                    disabled={!formValues.grad_opcina || availableNaselja.length === 0}
                  >
                    <option value="">{availableNaselja.length === 0 ? 'Nema dostupnih naselja' : 'Odaberite naselje'}</option>
                    {availableNaselja.map(n => (
                      <option key={n} value={n}>{n}</option>
                    ))}
                  </select>
                  {formValues.grad_opcina && availableNaselja.length === 0 && (
                    <p className="text-xs text-[var(--muted-foreground)] mt-1">Nema dodatnih naselja za ovaj grad</p>
                  )}
                </div>

                {/* Povrsina */}
                <div>
                  <label className="label">Stambena površina (m²)</label>
                  <input
                    type="number"
                    className="input"
                    placeholder="npr. 65"
                    value={formValues.stambena_povrsina}
                    onChange={(e) => setFormValues({...formValues, stambena_povrsina: e.target.value})}
                  />
                </div>

                {/* Broj soba */}
                <div>
                  <label className="label">Broj soba</label>
                  <select
                    className="select"
                    value={formValues.broj_soba}
                    onChange={(e) => setFormValues({...formValues, broj_soba: e.target.value})}
                  >
                    <option value="">Odaberite</option>
                    {propertyType === 'apartments' ? (
                      <>
                        <option value="0.5">Garsonijera</option>
                        <option value="1">1-sobni</option>
                        <option value="2">2-sobni</option>
                        <option value="3">3-sobni</option>
                        <option value="4">4-sobni</option>
                        <option value="5">5+ sobni</option>
                      </>
                    ) : (
                      <>
                        <option value="2">2</option>
                        <option value="3">3</option>
                        <option value="4">4</option>
                        <option value="5">5</option>
                        <option value="6">6</option>
                        <option value="7">7+</option>
                      </>
                    )}
                  </select>
                </div>

                {/* Godina izgradnje */}
                <div>
                  <label className="label">Godina izgradnje</label>
                  <input
                    type="number"
                    className="input"
                    placeholder="npr. 2010"
                    value={formValues.godina_izgradnje}
                    onChange={(e) => setFormValues({...formValues, godina_izgradnje: e.target.value})}
                  />
                </div>

                {/* Kat (only for apartments) */}
                {propertyType === 'apartments' && (
                  <div>
                    <label className="label">Kat</label>
                    <select
                      className="select"
                      value={formValues.kat}
                      onChange={(e) => setFormValues({...formValues, kat: e.target.value})}
                    >
                      <option value="">Odaberite</option>
                      <option value="-1">Suteren</option>
                      <option value="0">Prizemlje</option>
                      <option value="1">1. kat</option>
                      <option value="2">2. kat</option>
                      <option value="3">3. kat</option>
                      <option value="4">4. kat</option>
                      <option value="5">5+ kat</option>
                    </select>
                  </div>
                )}

                {/* Energetski razred */}
                <div>
                  <label className="label">Energetski razred</label>
                  <select
                    className="select"
                    value={formValues.energetski_razred}
                    onChange={(e) => setFormValues({...formValues, energetski_razred: e.target.value})}
                  >
                    <option value="">Odaberite</option>
                    <option value="5">A+</option>
                    <option value="4">A</option>
                    <option value="3">B</option>
                    <option value="2">C</option>
                    <option value="1">D</option>
                    <option value="0">E</option>
                    <option value="-1">F</option>
                    <option value="-2">G</option>
                  </select>
                </div>

                {/* Godina renovacije */}
                <div>
                  <label className="label">Godina renovacije</label>
                  <input
                    type="number"
                    className="input"
                    placeholder="npr. 2020"
                    value={formValues.godina_renovacije}
                    onChange={(e) => setFormValues({...formValues, godina_renovacije: e.target.value})}
                  />
                </div>

                {/* Kupaonice */}
                <div>
                  <label className="label">WC (bez tuša/kade)</label>
                  <select
                    className="select"
                    value={formValues.wc_broj}
                    onChange={(e) => setFormValues({...formValues, wc_broj: e.target.value})}
                  >
                    <option value="">Odaberite</option>
                    <option value="0">0</option>
                    <option value="1">1</option>
                    <option value="2">2</option>
                    <option value="3">3+</option>
                  </select>
                </div>

                <div>
                  <label className="label">Kupaonica s WC-om</label>
                  <select
                    className="select"
                    value={formValues.kupaonica_s_wc_broj}
                    onChange={(e) => setFormValues({...formValues, kupaonica_s_wc_broj: e.target.value})}
                  >
                    <option value="">Odaberite</option>
                    <option value="0">0</option>
                    <option value="1">1</option>
                    <option value="2">2</option>
                    <option value="3">3+</option>
                  </select>
                </div>

                {/* Broj etaza */}
                <div>
                  <label className="label">Broj etaža</label>
                  <select
                    className="select"
                    value={formValues.broj_etaza}
                    onChange={(e) => setFormValues({...formValues, broj_etaza: e.target.value})}
                  >
                    <option value="">Odaberite</option>
                    <option value="1">Jednoetažan</option>
                    <option value="2">Dvoetažan</option>
                    <option value="3">Višeetažan</option>
                  </select>
                </div>

                {/* Broj parkirnih mjesta */}
                <div>
                  <label className="label">Broj parkirnih mjesta</label>
                  <select
                    className="select"
                    value={formValues.broj_parkirnih_mjesta}
                    onChange={(e) => setFormValues({...formValues, broj_parkirnih_mjesta: e.target.value})}
                  >
                    <option value="">Odaberite</option>
                    <option value="0">0</option>
                    <option value="1">1</option>
                    <option value="2">2</option>
                    <option value="3">3+</option>
                  </select>
                </div>

                {/* Apartments: ukupni broj katova */}
                {propertyType === 'apartments' && (
                  <div>
                    <label className="label">Ukupno katova u zgradi</label>
                    <select
                      className="select"
                      value={formValues.ukupni_broj_katova}
                      onChange={(e) => setFormValues({...formValues, ukupni_broj_katova: e.target.value})}
                    >
                      <option value="">Odaberite</option>
                      {[...Array(15)].map((_, i) => (
                        <option key={i + 1} value={i + 1}>{i + 1}</option>
                      ))}
                      <option value="15">15+</option>
                    </select>
                  </div>
                )}

                {/* Apartments: tip stana */}
                {propertyType === 'apartments' && (
                  <div>
                    <label className="label">Tip stana</label>
                    <select
                      className="select"
                      value={formValues.tip_stana}
                      onChange={(e) => setFormValues({...formValues, tip_stana: e.target.value})}
                    >
                      <option value="">Odaberite</option>
                      <option value="u_stambenoj_zgradi">U stambenoj zgradi</option>
                      <option value="u_kući">U kući</option>
                    </select>
                  </div>
                )}

                {/* Apartments: orijentacija */}
                {propertyType === 'apartments' && (
                  <div>
                    <label className="label">Orijentacija</label>
                    <div className="flex flex-wrap gap-3 mt-1">
                      <label className="flex items-center gap-1.5 cursor-pointer text-sm">
                        <input type="checkbox" checked={formValues.orijentacija_sjever} onChange={(e) => setFormValues({...formValues, orijentacija_sjever: e.target.checked})} className="checkbox" />
                        Sjever
                      </label>
                      <label className="flex items-center gap-1.5 cursor-pointer text-sm">
                        <input type="checkbox" checked={formValues.orijentacija_jug} onChange={(e) => setFormValues({...formValues, orijentacija_jug: e.target.checked})} className="checkbox" />
                        Jug
                      </label>
                      <label className="flex items-center gap-1.5 cursor-pointer text-sm">
                        <input type="checkbox" checked={formValues.orijentacija_istok} onChange={(e) => setFormValues({...formValues, orijentacija_istok: e.target.checked})} className="checkbox" />
                        Istok
                      </label>
                      <label className="flex items-center gap-1.5 cursor-pointer text-sm">
                        <input type="checkbox" checked={formValues.orijentacija_zapad} onChange={(e) => setFormValues({...formValues, orijentacija_zapad: e.target.checked})} className="checkbox" />
                        Zapad
                      </label>
                    </div>
                  </div>
                )}

                {/* Houses: povrsina okucnice */}
                {propertyType === 'houses' && (
                  <div>
                    <label className="label">Površina okućnice (m²)</label>
                    <input
                      type="number"
                      className="input"
                      placeholder="npr. 500"
                      value={formValues.povrsina_okucnice}
                      onChange={(e) => setFormValues({...formValues, povrsina_okucnice: e.target.value})}
                    />
                  </div>
                )}

                {/* Houses: tip kuce */}
                {propertyType === 'houses' && (
                  <div>
                    <label className="label">Tip kuće</label>
                    <select
                      className="select"
                      value={formValues.tip_kuce}
                      onChange={(e) => setFormValues({...formValues, tip_kuce: e.target.value})}
                    >
                      <option value="">Odaberite</option>
                      <option value="samostojeća">Samostojeća</option>
                      <option value="dvojna_duplex">Dvojna / Duplex</option>
                      <option value="u_nizu">U nizu</option>
                      <option value="stambeno_poslovna">Stambeno-poslovna</option>
                    </select>
                  </div>
                )}

                {/* Houses: vrsta gradnje */}
                {propertyType === 'houses' && (
                  <div>
                    <label className="label">Vrsta gradnje</label>
                    <select
                      className="select"
                      value={formValues.vrsta_gradnje}
                      onChange={(e) => setFormValues({...formValues, vrsta_gradnje: e.target.value})}
                    >
                      <option value="">Odaberite</option>
                      <option value="zidana_kuća_beton">Zidana / Beton</option>
                      <option value="opeka">Opeka</option>
                      <option value="kamena_kuća">Kamena kuća</option>
                      <option value="montažna_kuća">Montažna kuća</option>
                      <option value="drvena_kuća">Drvena kuća</option>
                    </select>
                  </div>
                )}
              </div>

              {/* Main Checkboxes */}
              <div className="mt-6 flex flex-wrap gap-4">
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={formValues.novogradnja}
                    onChange={(e) => setFormValues({...formValues, novogradnja: e.target.checked})}
                    className="checkbox"
                  />
                  <span className="text-sm">Novogradnja</span>
                </label>
                {propertyType === 'apartments' && (
                  <label className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={formValues.ima_lift}
                      onChange={(e) => setFormValues({...formValues, ima_lift: e.target.checked})}
                      className="checkbox"
                    />
                    <span className="text-sm">Lift</span>
                  </label>
                )}
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={formValues.balkon_balkon}
                    onChange={(e) => setFormValues({...formValues, balkon_balkon: e.target.checked})}
                    className="checkbox"
                  />
                  <span className="text-sm">Balkon</span>
                </label>
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={formValues.balkon_terasa}
                    onChange={(e) => setFormValues({...formValues, balkon_terasa: e.target.checked})}
                    className="checkbox"
                  />
                  <span className="text-sm">Terasa</span>
                </label>
                {propertyType === 'houses' && (
                  <label className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={formValues.pogled_more}
                      onChange={(e) => setFormValues({...formValues, pogled_more: e.target.checked})}
                      className="checkbox"
                    />
                    <span className="text-sm">Pogled na more</span>
                  </label>
                )}
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={formValues.dozvola_vlasnicki_list}
                    onChange={(e) => setFormValues({...formValues, dozvola_vlasnicki_list: e.target.checked})}
                    className="checkbox"
                  />
                  <span className="text-sm">Vlasnički list</span>
                </label>
              </div>

              {/* Collapsible Advanced Section */}
              <div className="mt-6">
                <button
                  type="button"
                  onClick={() => setShowAdvanced(!showAdvanced)}
                  className="flex items-center gap-2 text-sm text-blue-500 hover:text-blue-600"
                >
                  <span>{showAdvanced ? '▼' : '▶'}</span>
                  <span>Dodatne informacije (opcionalno)</span>
                </button>

                {showAdvanced && (
                  <div className="mt-4 p-4 bg-[var(--secondary)] rounded-lg">
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                      {/* Parking type */}
                      <div>
                        <label className="label">Vrsta parkinga</label>
                        <select
                          className="select"
                          value={formValues.parking_type}
                          onChange={(e) => setFormValues({...formValues, parking_type: e.target.value})}
                        >
                          <option value="">Nema / Nepoznato</option>
                          <option value="garaža">Garaža</option>
                          <option value="garažno_mjesto">Garažno mjesto</option>
                          <option value="vanjsko_natkriveno">Vanjsko natkriveno</option>
                          <option value="vanjsko_ne_natkriveno">Vanjsko nenatkriveno</option>
                          <option value="besplatni_javni">Besplatni javni</option>
                          <option value="naplatni_javni">Naplatni javni</option>
                        </select>
                      </div>

                      {/* Grijanje sustav */}
                      <div>
                        <label className="label">Sustav grijanja</label>
                        <select
                          className="select"
                          value={formValues.grijanje_sustav}
                          onChange={(e) => setFormValues({...formValues, grijanje_sustav: e.target.value})}
                        >
                          <option value="">Nepoznato</option>
                          <option value="etažno_plinsko_centralno">Etažno plinsko centralno</option>
                          <option value="gradska_toplana">Gradska toplana</option>
                          <option value="dizalica_topline">Dizalica topline</option>
                          <option value="grijalice_i_radijatori_na_struju">Grijalice na struju</option>
                          {propertyType === 'houses' && (
                            <>
                              <option value="kotlovnica_na_plin">Kotlovnica na plin</option>
                              <option value="kotlovnica_na_drva">Kotlovnica na drva</option>
                            </>
                          )}
                          <option value="peć_na_drva">Peć na drva</option>
                          <option value="peć_na_plin">Peć na plin</option>
                          <option value="nema_sustav_grijanja">Nema grijanja</option>
                        </select>
                      </div>

                      {/* Klima */}
                      <div className="flex items-end">
                        <label className="flex items-center gap-2 cursor-pointer">
                          <input
                            type="checkbox"
                            checked={formValues.grijanje_klima}
                            onChange={(e) => setFormValues({...formValues, grijanje_klima: e.target.checked})}
                            className="checkbox"
                          />
                          <span className="text-sm">Klima uređaj</span>
                        </label>
                      </div>
                    </div>

                    {/* More checkboxes */}
                    <div className="mt-4 flex flex-wrap gap-4">
                      <label className="flex items-center gap-2 cursor-pointer">
                        <input
                          type="checkbox"
                          checked={formValues.balkon_lodja}
                          onChange={(e) => setFormValues({...formValues, balkon_lodja: e.target.checked})}
                          className="checkbox"
                        />
                        <span className="text-sm">Lođa</span>
                      </label>
                      <label className="flex items-center gap-2 cursor-pointer">
                        <input
                          type="checkbox"
                          checked={formValues.objekt_bazen}
                          onChange={(e) => setFormValues({...formValues, objekt_bazen: e.target.checked})}
                          className="checkbox"
                        />
                        <span className="text-sm">Bazen</span>
                      </label>
                      <label className="flex items-center gap-2 cursor-pointer">
                        <input
                          type="checkbox"
                          checked={formValues.objekt_dvoriste}
                          onChange={(e) => setFormValues({...formValues, objekt_dvoriste: e.target.checked})}
                          className="checkbox"
                        />
                        <span className="text-sm">Dvorište / Vrt</span>
                      </label>
                      <label className="flex items-center gap-2 cursor-pointer">
                        <input
                          type="checkbox"
                          checked={formValues.objekt_podrum}
                          onChange={(e) => setFormValues({...formValues, objekt_podrum: e.target.checked})}
                          className="checkbox"
                        />
                        <span className="text-sm">Podrum</span>
                      </label>
                      <label className="flex items-center gap-2 cursor-pointer">
                        <input
                          type="checkbox"
                          checked={formValues.objekt_rostilj}
                          onChange={(e) => setFormValues({...formValues, objekt_rostilj: e.target.checked})}
                          className="checkbox"
                        />
                        <span className="text-sm">Roštilj</span>
                      </label>
                      <label className="flex items-center gap-2 cursor-pointer">
                        <input
                          type="checkbox"
                          checked={formValues.objekt_spremiste}
                          onChange={(e) => setFormValues({...formValues, objekt_spremiste: e.target.checked})}
                          className="checkbox"
                        />
                        <span className="text-sm">Spremište</span>
                      </label>
                      <label className="flex items-center gap-2 cursor-pointer">
                        <input
                          type="checkbox"
                          checked={formValues.objekt_vrtna_kucica}
                          onChange={(e) => setFormValues({...formValues, objekt_vrtna_kucica: e.target.checked})}
                          className="checkbox"
                        />
                        <span className="text-sm">Vrtna kućica</span>
                      </label>
                      <label className="flex items-center gap-2 cursor-pointer">
                        <input
                          type="checkbox"
                          checked={formValues.objekt_zimski_vrt}
                          onChange={(e) => setFormValues({...formValues, objekt_zimski_vrt: e.target.checked})}
                          className="checkbox"
                        />
                        <span className="text-sm">Zimski vrt</span>
                      </label>
                    </div>

                    {/* Dozvole */}
                    <div className="mt-4">
                      <p className="text-sm text-[var(--muted-foreground)] mb-2">Dozvole</p>
                      <div className="flex flex-wrap gap-4">
                        <label className="flex items-center gap-2 cursor-pointer">
                          <input
                            type="checkbox"
                            checked={formValues.dozvola_gradevinska}
                            onChange={(e) => setFormValues({...formValues, dozvola_gradevinska: e.target.checked})}
                            className="checkbox"
                          />
                          <span className="text-sm">Građevinska dozvola</span>
                        </label>
                        <label className="flex items-center gap-2 cursor-pointer">
                          <input
                            type="checkbox"
                            checked={formValues.dozvola_uporabna}
                            onChange={(e) => setFormValues({...formValues, dozvola_uporabna: e.target.checked})}
                            className="checkbox"
                          />
                          <span className="text-sm">Uporabna dozvola</span>
                        </label>
                      </div>
                    </div>

                    {/* Funk features */}
                    <div className="mt-4">
                      <p className="text-sm text-[var(--muted-foreground)] mb-2">Dodatna oprema</p>
                      <div className="flex flex-wrap gap-4">
                        <label className="flex items-center gap-2 cursor-pointer">
                          <input
                            type="checkbox"
                            checked={formValues.funk_kamin}
                            onChange={(e) => setFormValues({...formValues, funk_kamin: e.target.checked})}
                            className="checkbox"
                          />
                          <span className="text-sm">Kamin</span>
                        </label>
                        <label className="flex items-center gap-2 cursor-pointer">
                          <input
                            type="checkbox"
                            checked={formValues.funk_podno_grijanje}
                            onChange={(e) => setFormValues({...formValues, funk_podno_grijanje: e.target.checked})}
                            className="checkbox"
                          />
                          <span className="text-sm">Podno grijanje</span>
                        </label>
                        <label className="flex items-center gap-2 cursor-pointer">
                          <input
                            type="checkbox"
                            checked={formValues.funk_alarm}
                            onChange={(e) => setFormValues({...formValues, funk_alarm: e.target.checked})}
                            className="checkbox"
                          />
                          <span className="text-sm">Alarm</span>
                        </label>
                        <label className="flex items-center gap-2 cursor-pointer">
                          <input
                            type="checkbox"
                            checked={formValues.funk_sauna}
                            onChange={(e) => setFormValues({...formValues, funk_sauna: e.target.checked})}
                            className="checkbox"
                          />
                          <span className="text-sm">Sauna</span>
                        </label>
                      </div>
                    </div>

                    {/* Alt energija */}
                    <div className="mt-4">
                      <p className="text-sm text-[var(--muted-foreground)] mb-2">Alternativni izvori energije</p>
                      <div className="flex flex-wrap gap-4">
                        <label className="flex items-center gap-2 cursor-pointer">
                          <input
                            type="checkbox"
                            checked={formValues.alt_solarni_paneli}
                            onChange={(e) => setFormValues({...formValues, alt_solarni_paneli: e.target.checked})}
                            className="checkbox"
                          />
                          <span className="text-sm">Solarni paneli</span>
                        </label>
                        <label className="flex items-center gap-2 cursor-pointer">
                          <input
                            type="checkbox"
                            checked={formValues.alt_toplinske_pumpe}
                            onChange={(e) => setFormValues({...formValues, alt_toplinske_pumpe: e.target.checked})}
                            className="checkbox"
                          />
                          <span className="text-sm">Toplinske pumpe</span>
                        </label>
                      </div>
                    </div>
                  </div>
                )}
              </div>

              {/* Submit Button */}
              <div className="mt-8">
                {backendReady === false && (
                  <p className="text-sm text-amber-500 mb-2">
                    ⏳ Model se pokreće (može potrajati do 60 sekundi)...
                  </p>
                )}
                <button
                  className="btn-primary w-full md:w-auto flex items-center justify-center gap-2"
                  onClick={handlePredict}
                  disabled={predicting || !formValues.stambena_povrsina || !formValues.zupanija}
                >
                  {predicting ? (
                    <>
                      <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                      Izračunavam...
                    </>
                  ) : (
                    <>
                      <Calculator className="w-4 h-4" />
                      Izračunaj cijenu
                    </>
                  )}
                </button>
                {predictionError && (
                  <p className="mt-2 text-sm text-red-500">{predictionError}</p>
                )}
              </div>
            </div>

            {/* Prediction Result */}
            {prediction && (
              <div className="mt-8 animate-fade-in">
                <div className="card bg-gradient-to-br from-blue-950 to-purple-950 border-blue-800">
                  {/* Main price + price per m² */}
                  <div className="text-center mb-6">
                    <p className="text-sm text-slate-400 mb-2">Procijenjena cijena</p>
                    <p className="text-5xl font-bold text-blue-400">{formatPrice(prediction.price)}</p>
                    <p className="text-lg text-slate-300 mt-2">{formatNumber(prediction.pricePerM2)} €/m²</p>
                    <p className="text-sm text-slate-400 mt-1">
                      Raspon: {formatPrice(prediction.range.low)} - {formatPrice(prediction.range.high)}
                    </p>
                  </div>

                  {/* City comparisons - full width grid */}
                  {prediction.comparisons.length > 0 && (
                    <div>
                      <h4 className="font-semibold mb-3 flex items-center justify-center gap-2 text-slate-300 text-sm">
                        <MapPin className="w-4 h-4" />
                        Ista nekretnina u drugim gradovima
                      </h4>
                      <div className="flex flex-wrap justify-center gap-2">
                        {prediction.comparisons.map((comp) => {
                          const diff = ((comp.price - prediction.price) / prediction.price) * 100;
                          const isMore = diff > 0;
                          return (
                            <div key={comp.city} className="bg-slate-700/40 rounded-lg p-3 text-center w-[calc(33.333%-0.375rem)] min-w-[140px] border border-slate-600/30">
                              <p className="text-sm text-slate-400 truncate">{comp.city}</p>
                              <p className="font-semibold text-slate-100 mt-1">{formatPrice(comp.price)}</p>
                              <p className={`text-xs mt-0.5 ${isMore ? 'text-red-400' : 'text-green-400'}`}>
                                {isMore ? '+' : ''}{diff.toFixed(0)}%
                              </p>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  )}

                  <p className="text-xs text-center text-slate-500 mt-6">
                    <Info className="w-3 h-3 inline-block mr-1" />
                    {prediction.modelVersion && (
                      <span>Model {prediction.modelVersion.toUpperCase()} ({prediction.featuresUsed} značajki) • </span>
                    )}
                    Trenirano na {formatNumber(edaData?.apartments.total_count || 0)} stanova i {formatNumber(edaData?.houses.total_count || 0)} kuća
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>
      </section>

      {/* Map Section */}
      <section id="karta" className="py-6 lg:py-8">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-5">
            <h2 className="text-3xl font-bold mb-2">Istraži tržište po gradovima</h2>
            <p className="text-[var(--muted-foreground)]">
              Klikni na grad za detaljnu analizu tog područja
            </p>
          </div>

          {/* City legend */}
          <div className="flex flex-wrap gap-3 justify-center mb-6">
            {filteredCities.slice(0, 10).map((city) => (
              <button
                key={city.name}
                onClick={() => {
                  setSelectedCity(city.name === selectedCity ? null : city.name);
                  if (city.name !== selectedCity) {
                    setTimeout(() => {
                      document.getElementById('analiza')?.scrollIntoView({ behavior: 'smooth' });
                    }, 100);
                  }
                }}
                className={`px-3 py-1.5 rounded-full text-sm font-medium transition-all ${
                  selectedCity === city.name
                    ? 'bg-blue-600 text-white'
                    : 'bg-[var(--card)] text-[var(--foreground)] border border-[var(--border)] hover:border-blue-500'
                }`}
              >
                {city.name}
              </button>
            ))}
          </div>

          {selectedCity && (
            <div className="mb-6 flex items-center justify-center gap-4">
              <span className="text-sm text-[var(--muted-foreground)]">Prikazujem podatke za:</span>
              <span className="px-3 py-1 bg-blue-600 text-white rounded-full font-medium text-sm">
                {selectedCity}
              </span>
              <button
                className="text-sm text-blue-400 hover:underline"
                onClick={() => setSelectedCity(null)}
              >
                Prikaži sve
              </button>
            </div>
          )}

          <div className="overflow-hidden">
            <MapComponent
              cities={filteredCities}
              onCitySelect={(city) => setSelectedCity(city === selectedCity ? null : city)}
              selectedCity={selectedCity}
              propertyType={propertyType}
            />
          </div>
        </div>
      </section>

      {/* EDA Section */}
      <section id="analiza" className="py-6 lg:py-8">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-5">
            <h2 className="text-3xl font-bold mb-2">
              Analiza tržišta {selectedCity ? `- ${selectedCity}` : ''}
            </h2>
            <p className="text-[var(--muted-foreground)]">
              {propertyType === 'apartments' ? 'Stanovi' : 'Kuće'} -
              {selectedCity
                ? ` Podaci za odabrani grad`
                : ` ${formatNumber(currentData?.total_count || 0)} nekretnina u bazi`
              }
            </p>
            <div className="flex justify-center mt-3">
              <button
                onClick={() => setRemoveOutliers(!removeOutliers)}
                className={`flex items-center gap-2 px-4 py-2 rounded-full text-sm font-medium transition-all ${
                  removeOutliers
                    ? 'bg-blue-600 text-white'
                    : 'bg-[var(--card)] text-[var(--muted-foreground)] border border-[var(--border)] hover:border-blue-500'
                }`}
              >
                <div className={`w-8 h-4 rounded-full relative transition-colors ${removeOutliers ? 'bg-blue-400' : 'bg-[var(--border)]'}`}>
                  <div className={`absolute top-0.5 w-3 h-3 rounded-full bg-white transition-all ${removeOutliers ? 'left-4' : 'left-0.5'}`} />
                </div>
                Bez outliera (IQR)
              </button>
            </div>
          </div>

          {/* Stats Overview */}
          <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3 mb-6">
            <div className="stat-card">
              <div className="stat-value text-2xl">{formatNumber(currentData?.total_count || 0)}</div>
              <div className="stat-label">Ukupno</div>
            </div>
            <div className="stat-card">
              <div className="stat-value text-2xl">{formatPrice(currentData?.avg_price || 0)}</div>
              <div className="stat-label">Prosjek</div>
            </div>
            <div className="stat-card">
              <div className="stat-value text-2xl">{formatPrice(currentData?.median_price || 0)}</div>
              <div className="stat-label">Medijan</div>
            </div>
            <div className="stat-card">
              <div className="stat-value text-2xl">{formatNumber(currentData?.avg_area || 0)} m²</div>
              <div className="stat-label">Prosj. površina</div>
            </div>
            <div className="stat-card">
              <div className="stat-value text-2xl">{formatNumber(currentData?.avg_price_per_m2 || 0)} €</div>
              <div className="stat-label">Prosj. €/m²</div>
            </div>
            <div className="stat-card">
              <div className="stat-value text-2xl">{formatNumber(currentData?.median_price_per_m2 || 0)} €</div>
              <div className="stat-label">Medijan €/m²</div>
            </div>
          </div>

          {/* Charts Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-5">
            {/* Price by Zupanija */}
            {!selectedCity && (
              <div className="chart-container">
                <h3 className="chart-title flex items-center gap-2">
                  <Euro className="w-4 h-4 text-blue-500" />
                  Prosječne cijene po županijama
                </h3>
                <ResponsiveContainer width="100%" height={350}>
                  <BarChart
                    data={currentData?.by_zupanija?.slice().sort((a, b) => (b.price_mean || 0) - (a.price_mean || 0)).slice(0, 10)}
                    layout="vertical"
                    margin={{ left: 120, right: 20 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                    <XAxis type="number" tickFormatter={(v) => `${(v/1000).toFixed(0)}k €`} />
                    <YAxis type="category" dataKey="zupanija" tick={{ fontSize: 12 }} width={120} />
                    <Tooltip
                      formatter={(value) => formatPrice(value as number)}
                      {...tooltipStyle}
                    />
                    <Bar dataKey="price_mean" fill={CHART_COLORS.primary} radius={[0, 4, 4, 0]} name="Prosječna cijena" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}

            {/* Price Distribution */}
            <div className="chart-container">
              <h3 className="chart-title flex items-center gap-2">
                <BarChart3 className="w-4 h-4 text-purple-500" />
                Distribucija cijena
              </h3>
              <ResponsiveContainer width="100%" height={350}>
                <AreaChart data={currentData?.price_distribution?.slice(0, 15)}>
                  <defs>
                    <linearGradient id="priceGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor={CHART_COLORS.secondary} stopOpacity={0.8}/>
                      <stop offset="95%" stopColor={CHART_COLORS.secondary} stopOpacity={0.1}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                  <XAxis dataKey="min" tickFormatter={(v) => `${(v/1000).toFixed(0)}k`} tick={{ fontSize: 11 }} />
                  <YAxis />
                  <Tooltip
                    formatter={(value) => [value as number, 'Broj nekretnina']}
                    labelFormatter={(v) => `${formatPrice(v as number)} - ${formatPrice((v as number) + 50000)}`}
                    {...tooltipStyle}
                  />
                  <Area type="monotone" dataKey="count" stroke={CHART_COLORS.secondary} fill="url(#priceGradient)" />
                </AreaChart>
              </ResponsiveContainer>
            </div>

            {/* Scatter Plot */}
            <div className="chart-container">
              <h3 className="chart-title flex items-center gap-2">
                <Ruler className="w-4 h-4 text-cyan-500" />
                Površina vs Cijena
              </h3>
              <ResponsiveContainer width="100%" height={350}>
                <ScatterChart margin={{ bottom: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                  <XAxis type="number" dataKey="stambena_povrsina" name="Površina" unit=" m²" tick={{ fontSize: 11 }} domain={[0, 'auto']} />
                  <YAxis type="number" dataKey="cijena" name="Cijena" tickFormatter={(v) => `${(v/1000).toFixed(0)}k`} domain={[0, 'auto']} />
                  <Tooltip
                    contentStyle={{
                      background: 'var(--card)',
                      border: '1px solid var(--border)',
                      borderRadius: '8px'
                    }}
                    itemStyle={{ color: CHART_COLORS.tertiary }}
                    content={({ active, payload }) => {
                      if (active && payload && payload.length) {
                        const data = payload[0].payload;
                        return (
                          <div style={{ background: 'var(--card)', border: '1px solid var(--border)', borderRadius: '8px', padding: '8px 12px' }}>
                            <p style={{ color: 'var(--foreground)', margin: 0 }}>{data.stambena_povrsina} m²</p>
                            <p style={{ color: CHART_COLORS.tertiary, margin: 0 }}>{formatPrice(data.cijena)}</p>
                          </div>
                        );
                      }
                      return null;
                    }}
                  />
                  <Scatter data={currentData?.scatter?.length && currentData.scatter.length > 500
                    ? currentData.scatter.filter((_, i) => i % Math.ceil(currentData.scatter.length / 500) === 0)
                    : currentData?.scatter} fill={CHART_COLORS.tertiary} fillOpacity={0.6} />
                </ScatterChart>
              </ResponsiveContainer>
            </div>

            {/* Price per m2 Distribution */}
            <div className="chart-container">
              <h3 className="chart-title flex items-center gap-2">
                <TrendingUp className="w-4 h-4 text-green-500" />
                Distribucija cijene po m²
              </h3>
              <ResponsiveContainer width="100%" height={350}>
                <AreaChart data={currentData?.price_per_m2_distribution?.slice(0, 15)}>
                  <defs>
                    <linearGradient id="m2Gradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor={CHART_COLORS.success} stopOpacity={0.8}/>
                      <stop offset="95%" stopColor={CHART_COLORS.success} stopOpacity={0.1}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                  <XAxis dataKey="min" tickFormatter={(v) => `${v} €`} tick={{ fontSize: 11 }} />
                  <YAxis />
                  <Tooltip
                    formatter={(value) => [value as number, 'Broj nekretnina']}
                    labelFormatter={(v) => `${v} - ${(v as number) + 500} €/m²`}
                    {...tooltipStyle}
                  />
                  <Area type="monotone" dataKey="count" stroke={CHART_COLORS.success} fill="url(#m2Gradient)" />
                </AreaChart>
              </ResponsiveContainer>
            </div>

            {/* Area Distribution */}
            <div className="chart-container">
              <h3 className="chart-title flex items-center gap-2">
                <Ruler className="w-4 h-4 text-orange-500" />
                Distribucija površine
              </h3>
              <ResponsiveContainer width="100%" height={350}>
                <BarChart data={currentData?.area_distribution}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                  <XAxis dataKey="min" tickFormatter={(v) => `${v} m²`} tick={{ fontSize: 11 }} />
                  <YAxis />
                  <Tooltip
                    formatter={(value) => [value as number, 'Broj nekretnina']}
                    labelFormatter={(v) => `${v} - ${(v as number) + 20} m²`}
                    {...tooltipStyle}
                  />
                  <Bar dataKey="count" fill={CHART_COLORS.warning} radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Rooms Distribution */}
            <div className="chart-container">
              <h3 className="chart-title flex items-center gap-2">
                <DoorOpen className="w-4 h-4 text-pink-500" />
                Distribucija broja soba
              </h3>
              <ResponsiveContainer width="100%" height={350}>
                <BarChart data={currentData?.rooms_distribution}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                  <XAxis dataKey="rooms" tick={{ fontSize: 11 }} tickFormatter={(v) => v === 0.5 ? 'Gars.' : `${v}`} />
                  <YAxis />
                  <Tooltip
                    formatter={(value) => [value as number, 'Broj nekretnina']}
                    labelFormatter={(v) => v === 0.5 ? 'Garsonijera' : `${v} soba`}
                    {...tooltipStyle}
                  />
                  <Bar dataKey="count" fill="#ec4899" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Year Distribution */}
            {currentData?.year_distribution && currentData.year_distribution.length > 0 && (
              <div className="chart-container">
                <h3 className="chart-title flex items-center gap-2">
                  <Calendar className="w-4 h-4 text-indigo-500" />
                  Distribucija godine izgradnje
                </h3>
                <ResponsiveContainer width="100%" height={350}>
                  <BarChart data={currentData.year_distribution}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                    <XAxis dataKey="decade" tick={{ fontSize: 11 }} />
                    <YAxis />
                    <Tooltip
                      formatter={(value) => [value as number, 'Broj nekretnina']}
                      {...tooltipStyle}
                    />
                    <Bar dataKey="count" fill="#6366f1" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}

            {/* Price per m2 by Zupanija */}
            {!selectedCity && (
              <div className="chart-container">
                <h3 className="chart-title flex items-center gap-2">
                  <TrendingUp className="w-4 h-4 text-emerald-500" />
                  Cijena po m² po županijama
                </h3>
                <ResponsiveContainer width="100%" height={350}>
                  <BarChart
                    data={currentData?.by_zupanija?.slice().sort((a, b) => (b.price_per_m2 || 0) - (a.price_per_m2 || 0)).slice(0, 10)}
                    layout="vertical"
                    margin={{ left: 120, right: 20 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                    <XAxis type="number" tickFormatter={(v) => `${v.toLocaleString()} €`} />
                    <YAxis type="category" dataKey="zupanija" tick={{ fontSize: 12 }} width={120} />
                    <Tooltip
                      formatter={(value) => `${formatNumber(value as number)} €/m²`}
                      {...tooltipStyle}
                    />
                    <Bar dataKey="price_per_m2" fill={CHART_COLORS.success} radius={[0, 4, 4, 0]} name="€/m²" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}
          </div>

          {/* Top Cities Tables - only show main cities */}
          {!selectedCity && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Most Expensive */}
              <div className="card">
                <h3 className="font-semibold text-lg mb-4 flex items-center gap-2">
                  <TrendingUp className="w-5 h-5 text-red-500" />
                  Najskuplji gradovi
                </h3>
                <div className="space-y-3">
                  {mainCitiesData.expensive.map((city, i) => (
                    <div key={city.grad_opcina} className="flex items-center justify-between p-3 bg-[var(--secondary)] rounded-lg">
                      <div className="flex items-center gap-3">
                        <span className="w-6 h-6 bg-red-500/20 text-red-400 rounded-full flex items-center justify-center text-xs font-bold">
                          {i + 1}
                        </span>
                        <span className="font-medium">{city.grad_opcina}</span>
                      </div>
                      <div className="text-right">
                        <div className="font-semibold">{formatPrice(city.price_mean || 0)}</div>
                        <div className="text-xs text-[var(--muted-foreground)]">{city.count} oglasa</div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Most Affordable */}
              <div className="card">
                <h3 className="font-semibold text-lg mb-4 flex items-center gap-2">
                  <TrendingUp className="w-5 h-5 text-green-500 rotate-180" />
                  Najpovoljniji gradovi
                </h3>
                <div className="space-y-3">
                  {mainCitiesData.affordable.map((city, i) => (
                    <div key={city.grad_opcina} className="flex items-center justify-between p-3 bg-[var(--secondary)] rounded-lg">
                      <div className="flex items-center gap-3">
                        <span className="w-6 h-6 bg-green-500/20 text-green-400 rounded-full flex items-center justify-center text-xs font-bold">
                          {i + 1}
                        </span>
                        <span className="font-medium">{city.grad_opcina}</span>
                      </div>
                      <div className="text-right">
                        <div className="font-semibold">{formatPrice(city.price_mean || 0)}</div>
                        <div className="text-xs text-[var(--muted-foreground)]">{city.count} oglasa</div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>
      </section>
    </div>
  );
}
