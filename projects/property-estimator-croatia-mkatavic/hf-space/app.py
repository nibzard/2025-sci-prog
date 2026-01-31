"""
Croatian Property Price Estimator - FastAPI Backend
Serves V3 LightGBM models for apartments and houses price prediction.
"""

from contextlib import asynccontextmanager
from typing import Optional, Literal
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Current year for age calculations
CURRENT_YEAR = 2026

# Coastal counties for pogled_more_primorje interaction
COASTAL_ZUPANIJE = {
    "Istarska", "Primorsko-goranska", "Zadarska",
    "Splitsko-dalmatinska", "Dubrovačko-neretvanska"
}

# Global model storage
models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup."""
    models["apartments"] = joblib.load("models/apartments_model_v3.joblib")
    models["houses"] = joblib.load("models/houses_model_v3.joblib")
    print(f"Loaded apartments model: {len(models['apartments']['feature_names'])} features")
    print(f"Loaded houses model: {len(models['houses']['feature_names'])} features")
    yield
    models.clear()


app = FastAPI(
    title="Croatian Property Price Estimator",
    description="V3 LightGBM model serving for Croatian property price estimation",
    version="3.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionRequest(BaseModel):
    """Request schema for property price prediction."""
    property_type: Literal["apartments", "houses"]

    # Location (required)
    zupanija: str
    grad_opcina: Optional[str] = None
    naselje: Optional[str] = None

    # Core numerics
    stambena_povrsina: float = Field(..., gt=0, description="Living area in m²")
    broj_soba: Optional[float] = None
    godina_izgradnje: Optional[int] = Field(None, ge=1800, le=2030)
    godina_renovacije: Optional[int] = Field(None, ge=1900, le=2030)
    broj_parkirnih_mjesta: Optional[int] = Field(None, ge=0)
    broj_etaza: Optional[float] = Field(None, ge=1)
    energetski_razred: Optional[int] = Field(None, ge=-2, le=5, description="A+=5, A=4, B=3, C=2, D=1, E=0, F=-1, G=-2")
    wc_broj: Optional[int] = Field(None, ge=0)
    kupaonica_s_wc_broj: Optional[int] = Field(None, ge=0)

    # Apartments only
    kat: Optional[float] = None
    ukupni_broj_katova: Optional[float] = None
    tip_stana: Optional[str] = None  # "u_kući" or "u_stambenoj_zgradi"

    # Houses only
    povrsina_okucnice: Optional[float] = Field(None, ge=0)
    pogled_more: Optional[int] = Field(None, ge=0, le=1)
    tip_kuce: Optional[str] = None  # "samostojeća", "dvojna_duplex", "u_nizu", "stambeno_poslovna"
    vrsta_gradnje: Optional[str] = None  # "zidana_kuća_beton", "kamena_kuća", "montažna_kuća", "drvena_kuća", "opeka"

    # Binary flags (shared)
    podatak_novogradnja: Optional[int] = Field(None, ge=0, le=1)
    podatak_lift: Optional[int] = Field(None, ge=0, le=1)
    balkon_balkon: Optional[int] = Field(None, ge=0, le=1)
    balkon_terasa: Optional[int] = Field(None, ge=0, le=1)
    balkon_lodja_loggia: Optional[int] = Field(None, ge=0, le=1)
    grijanje_klima: Optional[int] = Field(None, ge=0, le=1)
    objekt_bazen: Optional[int] = Field(None, ge=0, le=1)
    objekt_dvoriste_vrt: Optional[int] = Field(None, ge=0, le=1)
    objekt_podrum: Optional[int] = Field(None, ge=0, le=1)
    objekt_rostilj: Optional[int] = Field(None, ge=0, le=1)
    objekt_spremiste: Optional[int] = Field(None, ge=0, le=1)
    objekt_vrtna_kucica: Optional[int] = Field(None, ge=0, le=1)
    objekt_zimski_vrt: Optional[int] = Field(None, ge=0, le=1)

    # Orientation (apartments only)
    orijentacija_istok: Optional[int] = Field(None, ge=0, le=1)
    orijentacija_jug: Optional[int] = Field(None, ge=0, le=1)
    orijentacija_sjever: Optional[int] = Field(None, ge=0, le=1)
    orijentacija_zapad: Optional[int] = Field(None, ge=0, le=1)

    # Funk features
    funk_kamin: Optional[int] = Field(None, ge=0, le=1)
    funk_podno_grijanje: Optional[int] = Field(None, ge=0, le=1)
    funk_alarm: Optional[int] = Field(None, ge=0, le=1)
    funk_sauna: Optional[int] = Field(None, ge=0, le=1)

    # Alt energija
    alt_solarni_paneli: Optional[int] = Field(None, ge=0, le=1)
    alt_toplinske_pumpe: Optional[int] = Field(None, ge=0, le=1)

    # Parking type
    parking_type: Optional[str] = None

    # Heating system
    grijanje_sustav: Optional[str] = None

    # Permits
    dozvola_vlasnicki_list: Optional[int] = Field(None, ge=0, le=1)
    dozvola_uporabna_dozvola: Optional[int] = Field(None, ge=0, le=1)
    dozvola_gradevinska_dozvola: Optional[int] = Field(None, ge=0, le=1)


class PredictionResponse(BaseModel):
    """Response schema for property price prediction."""
    predicted_price: int
    price_per_m2: int
    price_range: dict
    model_version: str
    features_used: int


def build_feature_vector(request: PredictionRequest, model_data: dict) -> list:
    """
    Build feature vector in exact order expected by the model.
    Missing values are set to NaN (LightGBM handles them natively).
    """
    feature_names = model_data["feature_names"]
    encodings = model_data["encodings"]
    property_type = request.property_type

    # Initialize all features to NaN
    features = {name: np.nan for name in feature_names}

    # === DIRECT NUMERIC MAPPINGS ===
    features["stambena_povrsina"] = request.stambena_povrsina

    if request.broj_soba is not None:
        features["broj_soba"] = request.broj_soba

    if request.godina_izgradnje is not None:
        features["godina_izgradnje"] = request.godina_izgradnje
        features["ima_godinu_izgradnje"] = 1
    else:
        features["ima_godinu_izgradnje"] = 0

    if request.godina_renovacije is not None:
        features["godina_renovacije"] = request.godina_renovacije

    if request.broj_parkirnih_mjesta is not None:
        features["broj_parkirnih_mjesta"] = request.broj_parkirnih_mjesta

    if request.broj_etaza is not None:
        features["broj_etaza"] = request.broj_etaza

    if request.energetski_razred is not None:
        features["energetski_razred"] = request.energetski_razred
        features["ima_energetski_razred"] = 1
    else:
        features["ima_energetski_razred"] = 0

    if request.wc_broj is not None:
        features["wc_broj"] = request.wc_broj

    if request.kupaonica_s_wc_broj is not None:
        features["kupaonica_s_wc_broj"] = request.kupaonica_s_wc_broj

    # === APARTMENTS SPECIFIC ===
    if property_type == "apartments":
        if request.kat is not None:
            features["kat"] = request.kat
            # Set kat type flags
            features["kat_suteren"] = 1 if request.kat == -1 else 0
            features["kat_prizemlje"] = 1 if request.kat == 0 else 0
            features["kat_potkrovlje"] = 0  # Would need explicit flag from user
            features["kat_penthouse"] = 0

        if request.ukupni_broj_katova is not None:
            features["ukupni_broj_katova"] = request.ukupni_broj_katova
            features["ima_ukupni_broj_katova"] = 1
        else:
            features["ima_ukupni_broj_katova"] = 0

        if request.tip_stana == "u_kući":
            features["tip_stana_u_kući"] = 1
            features["tip_stana_u_stambenoj_zgradi"] = 0
        elif request.tip_stana == "u_stambenoj_zgradi":
            features["tip_stana_u_kući"] = 0
            features["tip_stana_u_stambenoj_zgradi"] = 1

    # === HOUSES SPECIFIC ===
    if property_type == "houses":
        if request.povrsina_okucnice is not None:
            features["povrsina_okucnice"] = request.povrsina_okucnice

        if request.pogled_more is not None:
            features["pogled_more"] = request.pogled_more

        # House type one-hot
        if request.tip_kuce:
            tip_kuce_map = {
                "samostojeća": "tip_kuce_samostojeća",
                "dvojna_duplex": "tip_kuce_dvojna_duplex",
                "u_nizu": "tip_kuce_u_nizu",
                "stambeno_poslovna": "tip_kuce_stambeno_poslovna"
            }
            if request.tip_kuce in tip_kuce_map:
                features[tip_kuce_map[request.tip_kuce]] = 1

        # Construction type one-hot
        if request.vrsta_gradnje:
            gradnja_map = {
                "zidana_kuća_beton": "gradnja_zidana_kuća_beton",
                "kamena_kuća": "gradnja_kamena_kuća",
                "montažna_kuća": "gradnja_montažna_kuća",
                "drvena_kuća": "gradnja_drvena_kuća",
                "opeka": "gradnja_opeka"
            }
            if request.vrsta_gradnje in gradnja_map:
                features[gradnja_map[request.vrsta_gradnje]] = 1

    # === BINARY FLAGS ===
    if request.podatak_novogradnja is not None:
        features["podatak_novogradnja"] = request.podatak_novogradnja

    if request.podatak_lift is not None:
        features["podatak_lift"] = request.podatak_lift

    if request.balkon_balkon is not None:
        features["balkon_balkon"] = request.balkon_balkon

    if request.balkon_terasa is not None:
        features["balkon_terasa"] = request.balkon_terasa

    if request.balkon_lodja_loggia is not None:
        features["balkon_lođa_loggia"] = request.balkon_lodja_loggia

    if request.grijanje_klima is not None:
        features["grijanje_klima"] = request.grijanje_klima

    if request.objekt_bazen is not None:
        features["objekt_bazen"] = request.objekt_bazen

    if request.objekt_dvoriste_vrt is not None:
        features["objekt_dvorište_vrt"] = request.objekt_dvoriste_vrt

    if request.objekt_podrum is not None:
        features["objekt_podrum"] = request.objekt_podrum

    if request.objekt_rostilj is not None:
        features["objekt_roštilj"] = request.objekt_rostilj

    if request.objekt_spremiste is not None:
        features["objekt_spremište"] = request.objekt_spremiste

    if request.objekt_vrtna_kucica is not None:
        features["objekt_vrtna_kućica"] = request.objekt_vrtna_kucica

    if request.objekt_zimski_vrt is not None:
        features["objekt_zimski_vrt"] = request.objekt_zimski_vrt

    # === ORIENTATION (apartments only) ===
    if property_type == "apartments":
        if request.orijentacija_istok is not None:
            features["orijentacija_istok"] = request.orijentacija_istok

        if request.orijentacija_jug is not None:
            features["orijentacija_jug"] = request.orijentacija_jug

        if request.orijentacija_sjever is not None:
            features["orijentacija_sjever"] = request.orijentacija_sjever

        if request.orijentacija_zapad is not None:
            features["orijentacija_zapad"] = request.orijentacija_zapad

    # === FUNK FEATURES ===
    if request.funk_kamin is not None:
        features["funk_kamin"] = request.funk_kamin

    if request.funk_podno_grijanje is not None:
        features["funk_podno_grijanje"] = request.funk_podno_grijanje

    if request.funk_alarm is not None:
        features["funk_alarm"] = request.funk_alarm

    if request.funk_sauna is not None:
        features["funk_sauna"] = request.funk_sauna

    # === ALT ENERGIJA ===
    if request.alt_solarni_paneli is not None:
        features["alt_energija_solarni_paneli"] = request.alt_solarni_paneli

    if request.alt_toplinske_pumpe is not None:
        features["alt_energija_toplinske_pumpe"] = request.alt_toplinske_pumpe

    # === PARKING TYPE ONE-HOT ===
    if request.parking_type:
        parking_map = {
            "garaža": "parking_garaža",
            "garažno_mjesto": "parking_garažno_mjesto",
            "vanjsko_natkriveno": "parking_vanjsko_natkriveno_mjesto",
            "vanjsko_ne_natkriveno": "parking_vanjsko_ne_natkriveno_mjesto",
            "besplatni_javni": "parking_besplatni_javni_parking",
            "naplatni_javni": "parking_naplatni_javni_parking"
        }
        if request.parking_type in parking_map:
            features[parking_map[request.parking_type]] = 1

    # === HEATING SYSTEM ONE-HOT ===
    if request.grijanje_sustav:
        # Map user-friendly names to feature names
        grijanje_map_apartments = {
            "dizalica_topline": "grijanje_sustav_dizalica_topline",
            "etažno_centralno_na_struju": "grijanje_sustav_etažno_centralno_na_struju",
            "etažno_plinsko_centralno": "grijanje_sustav_etažno_plinsko_centralno",
            "gradska_toplana": "grijanje_sustav_gradska_toplana",
            "grijalice_i_radijatori_na_struju": "grijanje_sustav_grijalice_i_radijatori_na_struju",
            "nema_sustav_grijanja": "grijanje_sustav_nema_sustav_grijanja",
            "peć_na_brikete_pelete": "grijanje_sustav_peć_na_brikete_pelete",
            "peć_na_drva": "grijanje_sustav_peć_na_drva",
            "peć_na_kruta_goriva": "grijanje_sustav_peć_na_kruta_goriva",
            "peć_na_lož_ulje": "grijanje_sustav_peć_na_lož_ulje",
            "peć_na_plin": "grijanje_sustav_peć_na_plin",
            "sustav_grijanja": "grijanje_sustav_sustav_grijanja",
            "zajednička_kotlovnica": "grijanje_sustav_zajednička_kotlovnica",
            "klimatizacije_i_ventilacije": "grijanje_sustav_klimatizacije_i_ventilacije"
        }
        grijanje_map_houses = {
            "dizalica_topline": "grijanje_sustav_dizalica_topline",
            "etažno_centralno_na_struju": "grijanje_sustav_etažno_centralno_na_struju",
            "etažno_plinsko_centralno": "grijanje_sustav_etažno_plinsko_centralno",
            "gradska_toplana": "grijanje_sustav_gradska_toplana",
            "grijalice_i_radijatori_na_struju": "grijanje_sustav_grijalice_i_radijatori_na_struju",
            "kotlovnica_na_brikete_pelete": "grijanje_sustav_kotlovnica_na_brikete_pelete",
            "kotlovnica_na_drva": "grijanje_sustav_kotlovnica_na_drva",
            "kotlovnica_na_kruta_goriva": "grijanje_sustav_kotlovnica_na_kruta_goriva",
            "kotlovnica_na_lož_ulje": "grijanje_sustav_kotlovnica_na_lož_ulje",
            "kotlovnica_na_plin": "grijanje_sustav_kotlovnica_na_plin",
            "nema_sustav_grijanja": "grijanje_sustav_nema_sustav_grijanja",
            "peć_na_brikete_pelete": "grijanje_sustav_peć_na_brikete_pelete",
            "peć_na_drva": "grijanje_sustav_peć_na_drva",
            "peć_na_kruta_goriva": "grijanje_sustav_peć_na_kruta_goriva",
            "peć_na_lož_ulje": "grijanje_sustav_peć_na_lož_ulje",
            "peć_na_plin": "grijanje_sustav_peć_na_plin",
            "sustav_grijanja": "grijanje_sustav_sustav_grijanja",
            "klimatizacije_i_ventilacije": "grijanje_sustav_klimatizacije_i_ventilacije"
        }
        grijanje_map = grijanje_map_houses if property_type == "houses" else grijanje_map_apartments
        if request.grijanje_sustav in grijanje_map:
            feature_name = grijanje_map[request.grijanje_sustav]
            if feature_name in features:
                features[feature_name] = 1
                features["ima_sustav_grijanja"] = 1

    # === PERMITS ===
    if request.dozvola_vlasnicki_list is not None:
        features["dozvola_vlasnički_list"] = request.dozvola_vlasnicki_list

    if request.dozvola_uporabna_dozvola is not None:
        features["dozvola_uporabna_dozvola"] = request.dozvola_uporabna_dozvola

    if request.dozvola_gradevinska_dozvola is not None:
        features["dozvola_građevinska_dozvola"] = request.dozvola_gradevinska_dozvola

    # === ENGINEERED FEATURES ===

    # starost (age of property)
    if request.godina_izgradnje is not None:
        starost = CURRENT_YEAR - request.godina_izgradnje
        features["starost"] = max(0, starost)

    # godine_od_renovacije (years since renovation)
    if request.godina_renovacije is not None:
        # Only count as renovation if different from build year
        if request.godina_izgradnje is None or request.godina_renovacije != request.godina_izgradnje:
            features["godine_od_renovacije"] = max(0, CURRENT_YEAR - request.godina_renovacije)
            features["ima_renovaciju"] = 1
        else:
            features["ima_renovaciju"] = 0
    else:
        features["ima_renovaciju"] = 0

    # povrsina_log (log of area)
    features["povrsina_log"] = np.log1p(request.stambena_povrsina)

    # ukupno_kupaonica (total bathrooms)
    wc = request.wc_broj if request.wc_broj is not None else 0
    kup = request.kupaonica_s_wc_broj if request.kupaonica_s_wc_broj is not None else 0
    features["ukupno_kupaonica"] = wc + kup

    # === INTERACTION FEATURES ===
    if property_type == "apartments":
        # lift_visoki_kat: lift is more valuable on higher floors
        lift = request.podatak_lift if request.podatak_lift is not None else 0
        kat = request.kat if request.kat is not None else 0
        features["lift_visoki_kat"] = lift * (1 if kat > 3 else 0)

        # novogradnja_povrsina: new construction * log(area)
        novo = request.podatak_novogradnja if request.podatak_novogradnja is not None else 0
        features["novogradnja_povrsina"] = novo * features["povrsina_log"]

    if property_type == "houses":
        # pogled_more_primorje: sea view more valuable in coastal regions
        pogled = request.pogled_more if request.pogled_more is not None else 0
        is_coastal = 1 if request.zupanija in COASTAL_ZUPANIJE else 0
        features["pogled_more_primorje"] = pogled * is_coastal

        # okucnica_log: log of plot size
        okucnica = request.povrsina_okucnice if request.povrsina_okucnice is not None else 0
        features["okucnica_log"] = np.log1p(okucnica)

        # bazen_uz_more: pool + sea view interaction
        bazen = request.objekt_bazen if request.objekt_bazen is not None else 0
        features["bazen_uz_more"] = bazen * pogled

    # === LOCATION ENCODING ===
    # Use target encodings from the trained model
    for loc_col in ["zupanija", "grad_opcina", "naselje"]:
        loc_value = getattr(request, loc_col, None)
        encoded_col = f"{loc_col}_encoded"

        if encoded_col in features:
            if loc_col in encodings and loc_value:
                mapping = encodings[loc_col]["mapping"]
                global_mean = encodings[loc_col]["global_mean"]
                features[encoded_col] = mapping.get(loc_value, global_mean)
            elif loc_col in encodings:
                features[encoded_col] = encodings[loc_col]["global_mean"]

    # Build final vector in exact feature order
    return [features.get(name, np.nan) for name in feature_names]


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "models_loaded": list(models.keys()),
        "apartments_features": len(models["apartments"]["feature_names"]) if "apartments" in models else 0,
        "houses_features": len(models["houses"]["feature_names"]) if "houses" in models else 0,
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Predict property price using V3 LightGBM model."""
    try:
        model_data = models[request.property_type]
        model = model_data["model"]

        # Build feature vector
        feature_vector = build_feature_vector(request, model_data)

        # Validate feature vector length
        expected_len = len(model_data["feature_names"])
        if len(feature_vector) != expected_len:
            raise HTTPException(
                status_code=500,
                detail=f"Feature vector length mismatch: got {len(feature_vector)}, expected {expected_len}"
            )

        # Predict in log space
        X = np.array([feature_vector])
        log_prediction = model.predict(X)[0]

        # Transform back to EUR (expm1 is inverse of log1p)
        predicted_price = float(np.expm1(log_prediction))
        predicted_price = max(0, predicted_price)

        # Calculate price per m2
        price_per_m2 = predicted_price / request.stambena_povrsina

        # Calculate confidence range based on model metrics
        # Using MAPE from test set as uncertainty estimate
        metrics = model_data.get("metrics", {})
        test_mape = metrics.get("test", {}).get("mape", 20) / 100  # Default 20% if not available

        price_range = {
            "low": int(predicted_price * (1 - test_mape)),
            "high": int(predicted_price * (1 + test_mape))
        }

        return PredictionResponse(
            predicted_price=int(predicted_price),
            price_per_m2=int(price_per_m2),
            price_range=price_range,
            model_version="v3",
            features_used=expected_len
        )

    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Invalid property type or missing data: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
