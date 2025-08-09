# src/index_graph/graph.py - VERSION COMPLÈTE AVEC IMAGES

import os
import re
import logging
from pathlib import Path
from typing import Optional, Any, Dict, List, Tuple
from datetime import datetime
import asyncio
import pandas as pd

from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document

from index_graph.configuration import IndexConfiguration
from index_graph.state import IndexState, InputState
from index_graph.pdf_visual_extractor import PDFVisualExtractor
from shared import retrieval
from shared.utils import sanitize_pinecone_id
from langgraph.graph import StateGraph, START, END

# Charger les variables d'environnement
load_dotenv()

# Configuration du logging
LOG_PATH = Path("indexing_errors.log")
logging.basicConfig(
    filename=LOG_PATH,
    filemode="a",
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)

# =============================================================================
# FONCTIONS UTILITAIRES POUR L'AMÉLIORATION DU TEXTE
# =============================================================================

def detect_ansd_content_type(text: str, pdf_name: str) -> str:
    """Détecte le type de contenu ANSD basé sur le texte et le nom du fichier."""
    text_lower = text.lower()
    pdf_lower = pdf_name.lower()
    
    content_patterns = {
        "rgph_demographics": ["population", "démographique", "ménage", "habitat", "rgph"],
        "rgph_economics": ["économie", "activité économique", "emploi", "chômage"],
        "rgph_education": ["éducation", "alphabétisation", "scolarisation", "instruction"],
        "rgph_health": ["santé", "handicap", "mortalité", "espérance de vie"],
        "rgph_marriage": ["matrimonial", "mariage", "union", "célibat"],
        "rgph_fertility": ["fécondité", "natalité", "naissances"],
        "rgph_migration": ["migration", "mobilité", "déplacement"],
        "rgph_methodology": ["méthodologie", "organisation", "enquête"],
        "statistics_table": ["tableau", "données", "statistiques", "résultats"],
        "executive_summary": ["synthèse", "résumé", "principales", "conclusions"]
    }
    
    # Vérifier le nom du fichier d'abord
    for content_type, keywords in content_patterns.items():
        if any(keyword in pdf_lower for keyword in keywords):
            return content_type
    
    # Vérifier le contenu du texte
    for content_type, keywords in content_patterns.items():
        keyword_count = sum(1 for keyword in keywords if keyword in text_lower)
        if keyword_count >= 2:
            return content_type
    
    return "rgph_general"


def extract_numerical_indicators(text: str) -> List[Dict[str, Any]]:
    """Extrait les indicateurs numériques importants du texte ANSD."""
    indicators = []
    
    demographic_patterns = [
        (r"population\s*(?:totale|du\s+sénégal)?\s*:\s*([0-9,\s]+(?:millions?|habitants?)?)", "population_totale"),
        (r"taux\s+de\s+croissance\s*:\s*([0-9,]+\s*%)", "taux_croissance"),
        (r"densité\s*:\s*([0-9,]+\s*(?:hab/km²|habitants?\s+par\s+km²)?)", "densite"),
        (r"espérance\s+de\s+vie\s*:\s*([0-9,]+\s*ans?)", "esperance_vie"),
        (r"taux\s+(?:d')?alphabétisation\s*:\s*([0-9,]+\s*%)", "alphabetisation"),
        (r"taux\s+de\s+scolarisation\s*:\s*([0-9,]+\s*%)", "scolarisation"),
        (r"(?:indice\s+synthétique\s+de\s+)?fécondité\s*:\s*([0-9,]+)", "fecondite"),
        (r"taux\s+de\s+mortalité\s+infantile\s*:\s*([0-9,]+\s*‰?)", "mortalite_infantile")
    ]
    
    for pattern, indicator_type in demographic_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            indicators.append({
                "type": indicator_type,
                "value": match.group(1).strip(),
                "context": text[max(0, match.start()-50):match.end()+50].strip(),
                "position": match.span()
            })
    
    return indicators


def clean_and_enhance_text(text: str, pdf_name: str) -> str:
    """Nettoie et enrichit le texte pour une meilleure indexation."""
    cleaned = re.sub(r'\s+', ' ', text)
    cleaned = re.sub(r'[^\w\s\-.,;:()%€£$°]', ' ', cleaned)
    
    # Corrections OCR
    ocr_corrections = {
        r'\bO\b': '0', r'\bl\b': '1', r'rn\b': 'm', r'\bS\b(?=\d)': '5',
    }
    
    for pattern, replacement in ocr_corrections.items():
        cleaned = re.sub(pattern, replacement, cleaned)
    
    content_type = detect_ansd_content_type(cleaned, pdf_name)
    
    type_context = {
        "rgph_demographics": "DÉMOGRAPHIE POPULATION",
        "rgph_economics": "ÉCONOMIE EMPLOI",
        "rgph_education": "ÉDUCATION ALPHABÉTISATION",
        "rgph_health": "SANTÉ HANDICAP",
        "rgph_marriage": "ÉTAT MATRIMONIAL",
        "rgph_fertility": "FÉCONDITÉ NATALITÉ",
        "rgph_migration": "MIGRATION MOBILITÉ",
        "rgph_methodology": "MÉTHODOLOGIE ENQUÊTE",
        "statistics_table": "DONNÉES STATISTIQUES",
        "executive_summary": "SYNTHÈSE RÉSULTATS"
    }
    
    context_prefix = type_context.get(content_type, "ANSD SÉNÉGAL")
    enhanced_text = f"[{context_prefix}] {cleaned}"
    
    return enhanced_text


def create_smart_text_splitter() -> RecursiveCharacterTextSplitter:
    """Crée un text splitter optimisé pour les documents ANSD."""
    ansd_separators = [
        "\n\n\n", "\n\n", "\n", ". ", "; ", ", ", " "
    ]
    
    return RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=300,
        separators=ansd_separators,
        length_function=len,
        is_separator_regex=False,
        keep_separator=True
    )


def extract_geographic_regions(text: str) -> List[str]:
    """Extrait les régions géographiques mentionnées dans le texte."""
    senegal_regions = [
        "dakar", "thiès", "saint-louis", "diourbel", "louga", "fatick", 
        "kaolack", "kolda", "ziguinchor", "tambacounda", "kaffrine", 
        "kédougou", "matam", "sédhiou", "rufisque", "pikine", "guédiawaye"
    ]
    
    found_regions = []
    text_lower = text.lower()
    
    for region in senegal_regions:
        if region in text_lower:
            found_regions.append(region.title())
    
    return found_regions


# =============================================================================
# FONCTIONS POUR L'INDEXATION VISUELLE AMÉLIORÉE
# =============================================================================

async def extract_text_from_image_content(image_path: Path, api_key: str) -> str:
    """Extrait le texte d'une image via l'API Vision OpenAI avec délais optimisés."""
    import base64
    import openai
    
    # Délai anti-rate-limit optimisé
    await asyncio.sleep(2.5)  # 2.5 secondes entre appels
    
    if not image_path.exists():
        print(f"         ⚠️ Image non trouvée: {image_path}")
        return ""
    
    try:
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        client = openai.OpenAI(api_key=api_key)
        
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": """Extrayez tout le texte visible dans cette image ANSD (graphique/tableau). 
                        Incluez titres, légendes, valeurs numériques, labels d'axes, unités.
                        Organisez le texte de manière structurée et logique.
                        Si c'est un graphique, décrivez les données principales.
                        Répondez en français."""
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}",
                            "detail": "high"
                        }
                    }
                ]
            }],
            max_tokens=500,
            temperature=0.1
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"         ❌ Erreur API Vision: {e}")
        return ""


async def extract_text_from_table_content(table_path: Path) -> str:
    """Extrait le texte d'un tableau CSV."""
    try:
        import pandas as pd
        
        # Lire le CSV avec différents encodages
        encodings = ['utf-8', 'latin-1', 'cp1252']
        df = None
        
        for encoding in encodings:
            try:
                df = pd.read_csv(table_path, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        
        if df is None:
            return ""
        
        # Nettoyer et formatter le contenu
        df = df.fillna('')  # Remplacer NaN par chaîne vide
        
        # Construire une représentation textuelle du tableau
        text_parts = []
        
        # En-têtes
        if not df.columns.empty:
            headers = " | ".join(str(col) for col in df.columns)
            text_parts.append(f"Colonnes: {headers}")
        
        # Données (limiter aux 20 premières lignes)
        max_rows = min(20, len(df))
        for idx, row in df.head(max_rows).iterrows():
            row_text = " | ".join(str(val) for val in row.values if str(val).strip())
            if row_text.strip():
                text_parts.append(f"Ligne {idx + 1}: {row_text}")
        
        if len(df) > max_rows:
            text_parts.append(f"... et {len(df) - max_rows} lignes supplémentaires")
        
        return "\n".join(text_parts)
        
    except Exception as e:
        logging.error(f"Erreur extraction tableau {table_path}: {e}")
        return ""


async def index_visual_content(vectorstore, cfg: IndexConfiguration) -> Dict[str, int]:
    """Indexe le contenu visuel (graphiques et tableaux) extrait - VERSION AMÉLIORÉE."""
    stats = {
        "charts_indexed": 0,
        "tables_indexed": 0,
        "charts_failed": 0,
        "tables_failed": 0,
        "total_visual_indicators": 0
    }
    
    # 1. Indexer les graphiques avec traitement intelligent
    charts_path = Path(cfg.chart_index_path)
    if charts_path.exists():
        try:
            charts_df = pd.read_csv(charts_path)
            print(f"   📊 Indexation intelligente de {len(charts_df)} graphiques...")
            
            batch_size = getattr(cfg, 'visual_batch_size', 2)  # Batch plus petit pour stabilité
            total_processed = 0
            
            for i in range(0, len(charts_df), batch_size):
                batch = charts_df.iloc[i:i+batch_size]
                texts, metadatas, ids = [], [], []
                
                print(f"      🔄 Traitement batch graphiques {i//batch_size + 1}/{(len(charts_df)-1)//batch_size + 1}")
                
                for idx, row in batch.iterrows():
                    try:
                        image_path = Path(row['image_path'])
                        
                        print(f"         📸 Extraction texte: {image_path.name}")
                        
                        # Extraire le texte de l'image
                        extracted_text = await extract_text_from_image_content(
                            image_path, cfg.api_key
                        )
                        
                        if extracted_text and len(extracted_text.strip()) > 10:
                            # Nettoyer et enrichir le texte extrait
                            enhanced_text = clean_and_enhance_text(extracted_text, image_path.name)
                            
                            # Extraire les indicateurs du contenu visuel
                            indicators = extract_numerical_indicators(enhanced_text)
                            stats["total_visual_indicators"] += len(indicators)
                            
                            # Détecter le type de graphique
                            chart_type = detect_chart_type(enhanced_text, image_path.name)
                            
                            # Créer le contenu du document enrichi
                            content_parts = [
                                f"Type: Graphique ANSD ({chart_type})",
                                f"Caption: {row.get('caption', 'Non spécifiée')}",
                                f"Source PDF: {Path(row.get('pdf_path', '')).name}",
                                f"Page: {row.get('page', 'N/A')}",
                                f"Contenu extrait: {enhanced_text}"
                            ]
                            
                            if indicators:
                                indicators_text = "; ".join([f"{ind['type']}: {ind['value']}" for ind in indicators])
                                content_parts.append(f"Indicateurs détectés: {indicators_text}")
                            
                            content = "\n".join(content_parts)
                            
                            # Métadonnées enrichies
                            metadata = {
                                "type": "visual_chart",
                                "source_type": "visual",
                                "chart_type": chart_type,
                                "image_id": str(row.get('image_id', f"chart_{idx}")),
                                "pdf_path": str(row.get('pdf_path', '')),
                                "pdf_name": Path(row.get('pdf_path', '')).name,
                                "page": int(row.get('page', 0)),
                                "image_path": str(image_path),
                                "caption": str(row.get('caption', '')),
                                "indexed_at": datetime.utcnow().isoformat(),
                                "content_source": "vision_api_extraction",
                                "content_length": len(extracted_text),
                                "indicators_count": len(indicators),
                                "document_source": "ansd_rgph",
                                "language": "french",
                                "country": "senegal"
                            }
                            
                            if indicators:
                                metadata["numerical_indicators"] = [
                                    {"type": ind["type"], "value": ind["value"]} 
                                    for ind in indicators
                                ]
                            
                            # ID sécurisé
                            raw_chart_id = f"visual_chart_{row.get('image_id', idx)}"
                            doc_id = sanitize_pinecone_id(raw_chart_id)
                            
                            texts.append(content)
                            metadatas.append(metadata)
                            ids.append(doc_id)
                            print(f"         ✅ Texte extrait: {len(extracted_text)} caractères")
                            
                        else:
                            print(f"         ⚠️ Texte vide ou trop court: {image_path.name}")
                            stats["charts_failed"] += 1
                            
                    except Exception as e:
                        logging.error(f"Erreur traitement graphique {idx}: {e}")
                        print(f"         ❌ Erreur: {e}")
                        stats["charts_failed"] += 1
                        continue
                
                # Indexer le batch
                if texts:
                    try:
                        await asyncio.to_thread(
                            vectorstore.add_texts,
                            texts=texts,
                            metadatas=metadatas,
                            ids=ids
                        )
                        stats["charts_indexed"] += len(texts)
                        total_processed += len(texts)
                        print(f"      ✅ Batch graphiques indexé: {len(texts)} éléments")
                    except Exception as e:
                        logging.error(f"Erreur indexation batch graphiques: {e}")
                        print(f"      ❌ Erreur indexation batch: {e}")
                        stats["charts_failed"] += len(texts)
                else:
                    print(f"      ⚠️ Aucun texte valide dans ce batch")
                
                # Pause pour éviter les limites API
                await asyncio.sleep(1.5)
            
            print(f"   📊 Graphiques traités: {total_processed}/{len(charts_df)}")
                        
        except Exception as e:
            logging.error(f"Erreur indexation graphiques: {e}")
            print(f"   ❌ Erreur générale graphiques: {e}")
    
    # 2. Indexer les tableaux
    tables_path = Path(cfg.table_index_path)
    if tables_path.exists():
        try:
            tables_df = pd.read_csv(tables_path)
            print(f"   📋 Indexation intelligente de {len(tables_df)} tableaux...")
            
            batch_size = getattr(cfg, 'visual_batch_size', 3)  # Tableaux plus rapides
            total_processed = 0
            
            for i in range(0, len(tables_df), batch_size):
                batch = tables_df.iloc[i:i+batch_size]
                texts, metadatas, ids = [], [], []
                
                print(f"      🔄 Traitement batch tableaux {i//batch_size + 1}/{(len(tables_df)-1)//batch_size + 1}")
                
                for idx, row in batch.iterrows():
                    try:
                        table_path = Path(row['table_path'])
                        
                        print(f"         📊 Extraction tableau: {table_path.name}")
                        
                        # Extraire le texte du tableau
                        extracted_text = await extract_text_from_table_content(table_path)
                        
                        if extracted_text and len(extracted_text.strip()) > 20:
                            # Enrichir le contenu du tableau
                            enhanced_text = clean_and_enhance_text(extracted_text, table_path.name)
                            
                            # Extraire les indicateurs
                            indicators = extract_numerical_indicators(enhanced_text)
                            stats["total_visual_indicators"] += len(indicators)
                            
                            # Créer le contenu du document
                            content_parts = [
                                f"Type: Tableau ANSD",
                                f"Caption: {row.get('caption', 'Non spécifiée')}",
                                f"Source PDF: {Path(row.get('pdf_path', '')).name}",
                                f"Page: {row.get('page', 'N/A')}",
                                f"Données du tableau: {enhanced_text}"
                            ]
                            
                            if indicators:
                                indicators_text = "; ".join([f"{ind['type']}: {ind['value']}" for ind in indicators])
                                content_parts.append(f"Indicateurs détectés: {indicators_text}")
                            
                            content = "\n".join(content_parts)
                            
                            # Métadonnées enrichies
                            metadata = {
                                "type": "visual_table",
                                "source_type": "visual",
                                "table_id": str(row.get('table_id', f"table_{idx}")),
                                "pdf_path": str(row.get('pdf_path', '')),
                                "pdf_name": Path(row.get('pdf_path', '')).name,
                                "page": int(row.get('page', 0)),
                                "table_path": str(table_path),
                                "caption": str(row.get('caption', '')),
                                "indexed_at": datetime.utcnow().isoformat(),
                                "content_source": "csv_parsing",
                                "content_length": len(extracted_text),
                                "indicators_count": len(indicators),
                                "document_source": "ansd_rgph",
                                "language": "french",
                                "country": "senegal"
                            }
                            
                            if indicators:
                                metadata["numerical_indicators"] = [
                                    {"type": ind["type"], "value": ind["value"]} 
                                    for ind in indicators
                                ]
                            
                            # ID sécurisé
                            raw_table_id = f"visual_table_{row.get('table_id', idx)}"
                            doc_id = sanitize_pinecone_id(raw_table_id)
                            
                            texts.append(content)
                            metadatas.append(metadata)
                            ids.append(doc_id)
                            print(f"         ✅ Tableau traité: {len(extracted_text)} caractères")
                            
                        else:
                            print(f"         ⚠️ Tableau vide ou trop petit: {table_path.name}")
                            stats["tables_failed"] += 1
                            
                    except Exception as e:
                        logging.error(f"Erreur traitement tableau {idx}: {e}")
                        print(f"         ❌ Erreur: {e}")
                        stats["tables_failed"] += 1
                        continue
                
                # Indexer le batch
                if texts:
                    try:
                        await asyncio.to_thread(
                            vectorstore.add_texts,
                            texts=texts,
                            metadatas=metadatas,
                            ids=ids
                        )
                        stats["tables_indexed"] += len(texts)
                        total_processed += len(texts)
                        print(f"      ✅ Batch tableaux indexé: {len(texts)} éléments")
                    except Exception as e:
                        logging.error(f"Erreur indexation batch tableaux: {e}")
                        print(f"      ❌ Erreur indexation batch: {e}")
                        stats["tables_failed"] += len(texts)
                else:
                    print(f"      ⚠️ Aucun contenu valide dans ce batch")
                
                # Petite pause
                await asyncio.sleep(0.5)
            
            print(f"   📋 Tableaux traités: {total_processed}/{len(tables_df)}")
                        
        except Exception as e:
            logging.error(f"Erreur indexation tableaux: {e}")
            print(f"   ❌ Erreur générale tableaux: {e}")
    
    return stats


def detect_chart_type(text: str, filename: str) -> str:
    """Détecte le type de graphique basé sur le contenu."""
    text_lower = text.lower()
    filename_lower = filename.lower()
    
    chart_patterns = {
        "pyramide_ages": ["pyramide", "âges", "population par âge"],
        "evolution_temporelle": ["évolution", "tendance", "années", "croissance"],
        "repartition": ["répartition", "pourcentage", "distribution"],
        "comparaison_regionale": ["région", "régional", "départements"],
        "indicateur_demographique": ["taux", "densité", "mortalité", "natalité"],
        "graphique_economique": ["emploi", "activité", "secteur", "économique"],
        "carte_geographique": ["carte", "géographique", "localisation"]
    }
    
    # Vérifier le nom du fichier
    for chart_type, keywords in chart_patterns.items():
        if any(keyword in filename_lower for keyword in keywords):
            return chart_type
    
    # Vérifier le contenu
    for chart_type, keywords in chart_patterns.items():
        if any(keyword in text_lower for keyword in keywords):
            return chart_type
    
    return "graphique_general"


# =============================================================================
# FONCTIONS PRINCIPALES
# =============================================================================

def check_index_config(
    state: IndexState,
    *,
    config: Optional[RunnableConfig] = None
) -> Dict[str, Any]:
    """Valide la configuration et le répertoire PDF."""
    configuration = IndexConfiguration.from_runnable_config(config)

    if not configuration.api_key:
        raise ValueError("API key is required for document indexing.")

    # Validation optionnelle de l'INDEX_API_KEY pour plus de flexibilité
    # if configuration.api_key != os.getenv("INDEX_API_KEY"):
    #     raise ValueError("Authentication failed: Invalid API key provided.")

    if configuration.retriever_provider != "pinecone":
        raise ValueError(
            "Only Pinecone is currently supported for document indexing."
        )

    if not state.pdf_root:
        raise ValueError("pdf_root doit être spécifié et non vide")

    pdf_root = Path(state.pdf_root).expanduser().resolve()
    if not pdf_root.exists():
        raise FileNotFoundError(f"Répertoire PDF introuvable : {pdf_root}")
    
    if not pdf_root.is_dir():
        raise ValueError(f"pdf_root doit être un répertoire : {pdf_root}")

    pdf_files = list(pdf_root.glob("**/*.pdf"))
    total_size = sum(f.stat().st_size for f in pdf_files if f.exists())
    
    logging.info(f"📁 Répertoire validé: {pdf_root}")
    logging.info(f"📄 {len(pdf_files)} fichiers PDF trouvés")
    logging.info(f"💾 Taille totale: {total_size/1024/1024:.1f} MB")

    return {"total_files_found": len(pdf_files)}


async def extract_visual_content(
    state: IndexState,
    *,
    config: Optional[RunnableConfig] = None
) -> Dict[str, Any]:
    """Étape 1: Extraire les graphiques et tableaux des PDFs."""
    print("🔍 ÉTAPE 1: Extraction du contenu visuel des PDFs...")
    
    cfg = IndexConfiguration.from_runnable_config(config)
    pdf_root = Path(state.pdf_root).expanduser().resolve()
    
    if not cfg.enable_visual_indexing:
        print("⚠️ Extraction visuelle désactivée dans la configuration")
        return {
            "visual_extraction_completed": True,
            "total_images_extracted": 0,
            "total_tables_extracted": 0,
            "extraction_errors": ["Extraction visuelle désactivée dans la configuration"]
        }
    
    try:
        extractor = PDFVisualExtractor(
            output_dir=cfg.output_dir if hasattr(cfg, 'output_dir') else ".",
            images_dir=cfg.images_dir,
            tables_dir=cfg.tables_dir
        )
        
        print(f"📁 Extraction depuis: {pdf_root}")
        
        extraction_result = await asyncio.to_thread(
            extractor.process_directory,
            pdf_root
        )
        
        print(f"✅ Extraction terminée:")
        print(f"   📊 {extraction_result['total_images']} images extraites")
        print(f"   📋 {extraction_result['total_tables']} tableaux extraits")
        print(f"   📄 {extraction_result['processed_pdfs']} PDFs traités")
        
        return {
            "visual_extraction_completed": True,
            "total_images_extracted": extraction_result["total_images"],
            "total_tables_extracted": extraction_result["total_tables"],
            "extraction_errors": extraction_result.get("processing_errors", [])
        }
        
    except Exception as e:
        error_msg = f"Erreur lors de l'extraction visuelle: {e}"
        logging.error(error_msg)
        print(f"❌ {error_msg}")
        
        return {
            "visual_extraction_completed": False,
            "total_images_extracted": 0,
            "total_tables_extracted": 0,
            "extraction_errors": [error_msg]
        }


async def index_local_pdfs(
    state: IndexState,
    *,
    config: Optional[RunnableConfig] = None
) -> Dict[str, Any]:
    """Indexe tous les PDFs avec traitement intelligent du texte ET des images."""
    print("📚 ÉTAPE 2: Indexation du contenu textuel et visuel...")
    
    cfg = IndexConfiguration.from_runnable_config(config)
    pdf_root = Path(state.pdf_root).expanduser().resolve()
    pdf_paths = list(pdf_root.glob("**/*.pdf"))
    
    stats = {
        "processed_files": [],
        "failed_files": [],
        "total_text_chunks": 0,
        "total_visual_content": 0,
        "total_indicators_extracted": 0,
        "content_types_found": set(),
        "regions_mentioned": set(),
        "visual_indexing_stats": {
            "charts_indexed": 0,
            "tables_indexed": 0,
            "charts_failed": 0,
            "tables_failed": 0,
            "total_visual_indicators": 0
        },
        "status": "Indexation intelligente complète en cours...",
        "total_files_found": len(pdf_paths)
    }

    if not pdf_paths:
        return {
            **stats,
            "status": "Aucun fichier PDF trouvé dans le répertoire"
        }

    logging.info(f"🚀 Début de l'indexation complète de {len(pdf_paths)} PDFs")
    now_str = datetime.utcnow().isoformat()
    
    async with retrieval.make_retriever(config) as retriever:
        vectorstore = retriever.vectorstore
        
        # 1. Indexation du contenu textuel
        print("📄 Indexation du contenu textuel...")
        text_splitter = create_smart_text_splitter()
        all_texts, all_metadatas, all_ids = [], [], []

        for idx, pdf_path in enumerate(pdf_paths, 1):
            try:
                logging.info(f"📖 ({idx}/{len(pdf_paths)}) Traitement textuel: {pdf_path.name}")
                
                loader = PyPDFLoader(str(pdf_path))
                pages = await asyncio.to_thread(loader.load_and_split)
                
                file_chunks = 0
                file_indicators = 0
                
                for page_doc in pages:
                    raw_text = page_doc.page_content.strip()
                    if not raw_text or len(raw_text) < 50:
                        continue

                    enhanced_text = clean_and_enhance_text(raw_text, pdf_path.name)
                    indicators = extract_numerical_indicators(enhanced_text)
                    file_indicators += len(indicators)
                    
                    regions = extract_geographic_regions(enhanced_text)
                    stats["regions_mentioned"].update(regions)
                    
                    content_type = detect_ansd_content_type(enhanced_text, pdf_path.name)
                    stats["content_types_found"].add(content_type)
                    
                    chunks = text_splitter.split_text(enhanced_text)
                    page_num = page_doc.metadata.get("page", 0)
                    
                    for chunk_idx, chunk_text in enumerate(chunks):
                        raw_id = f"{pdf_path.stem}_p{page_num}_c{chunk_idx}"
                        vector_id = sanitize_pinecone_id(raw_id)

                        metadata = {
                            "page_num": page_num,
                            "chunk_idx": chunk_idx,
                            "pdf_path": str(pdf_path),
                            "pdf_name": pdf_path.name,
                            "pdf_dir": str(pdf_path.parent),
                            "file_size": pdf_path.stat().st_size,
                            "indexed_at": now_str,
                            "content_type": content_type,
                            "chunk_length": len(chunk_text),
                            "regions_mentioned": list(regions),
                            "indicators_count": len([ind for ind in indicators if chunk_text in ind["context"]]),
                            "document_source": "ansd_rgph",
                            "language": "french",
                            "country": "senegal",
                            "type": "text_content"
                        }
                        
                        chunk_indicators = [
                            ind for ind in indicators 
                            if any(word in chunk_text.lower() for word in ind["context"].lower().split())
                        ]
                        
                        if chunk_indicators:
                            metadata["numerical_indicators"] = [
                                {"type": ind["type"], "value": ind["value"]} 
                                for ind in chunk_indicators
                            ]

                        all_texts.append(chunk_text)
                        all_metadatas.append(metadata)
                        all_ids.append(vector_id)
                        file_chunks += 1
                
                if file_chunks > 0:
                    stats["processed_files"].append(pdf_path.name)
                    stats["total_indicators_extracted"] += file_indicators
                    logging.info(f"✅ {pdf_path.name}: {file_chunks} chunks, {file_indicators} indicateurs")
                        
            except Exception as e:
                logging.error(f"❌ Erreur textuelle {pdf_path.name}: {e}")
                stats["failed_files"].append(pdf_path.name)
                continue

        # Indexation textuelle par lots
        if all_texts:
            batch_size = 25
            successfully_indexed = 0
            
            for i in range(0, len(all_texts), batch_size):
                batch_texts = all_texts[i:i + batch_size]
                batch_metadatas = all_metadatas[i:i + batch_size]
                batch_ids = all_ids[i:i + batch_size]
                
                try:
                    await asyncio.to_thread(
                        vectorstore.add_texts,
                        texts=batch_texts,
                        metadatas=batch_metadatas,
                        ids=batch_ids
                    )
                    successfully_indexed += len(batch_texts)
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    logging.error(f"Erreur indexation batch texte {i//batch_size + 1}: {e}")
                    continue
            
            stats["total_text_chunks"] = successfully_indexed
            print(f"✅ Contenu textuel indexé: {successfully_indexed} chunks")

        # 2. Indexation du contenu visuel SI activé ET extraction complétée
        visual_extraction_completed = getattr(state, 'visual_extraction_completed', False)
        
        if cfg.enable_visual_indexing and (Path(cfg.chart_index_path).exists() or Path(cfg.table_index_path).exists()):
            print("🎨 Indexation du contenu visuel...")
            
            visual_stats = await index_visual_content(
                vectorstore=vectorstore,
                cfg=cfg
            )
            
            stats["visual_indexing_stats"] = visual_stats
            stats["total_visual_content"] = (
                visual_stats["charts_indexed"] + visual_stats["tables_indexed"]
            )
            
            print(f"✅ Contenu visuel indexé:")
            print(f"   📊 {visual_stats['charts_indexed']} graphiques")
            print(f"   📋 {visual_stats['tables_indexed']} tableaux")
            print(f"   🔢 {visual_stats.get('total_visual_indicators', 0)} indicateurs visuels")
        
        else:
            if not cfg.enable_visual_indexing:
                print("⚠️ Indexation visuelle désactivée dans la configuration")
            else:
                print("⚠️ Indexation visuelle ignorée (fichiers d'index non trouvés)")

        # 3. Rapport final enrichi
        stats["content_types_found"] = list(stats["content_types_found"])
        stats["regions_mentioned"] = list(stats["regions_mentioned"])
        
        total_content = stats["total_text_chunks"] + stats["total_visual_content"]
        success_rate = len(stats["processed_files"]) / len(pdf_paths) * 100
        
        final_status = (
            f"Indexation complète terminée: {total_content} éléments "
            f"({stats['total_text_chunks']} texte + {stats['total_visual_content']} visuel) "
            f"depuis {len(stats['processed_files'])}/{len(pdf_paths)} PDFs "
            f"({success_rate:.1f}% succès)"
        )

        logging.info(f"🎉 {final_status}")
        logging.info(f"📊 Types de contenu: {', '.join(stats['content_types_found'])}")
        logging.info(f"🗺️ Régions mentionnées: {', '.join(stats['regions_mentioned'])}")
        
        return {
            **stats,
            "status": final_status
        }


# =============================================================================
# CONSTRUCTION DU GRAPHE COMPLET
# =============================================================================

builder = StateGraph(IndexState, input=InputState, config_schema=IndexConfiguration)

# Nœuds
builder.add_node("check_index_config", check_index_config)
builder.add_node("extract_visual_content", extract_visual_content)
builder.add_node("index_local_pdfs", index_local_pdfs)

# Flux d'exécution
builder.add_edge(START, "check_index_config")
builder.add_edge("check_index_config", "extract_visual_content")
builder.add_edge("extract_visual_content", "index_local_pdfs")
builder.add_edge("index_local_pdfs", END)

# Compilation
graph = builder.compile()
graph.name = "CompleteIntelligentANSDIndexGraph"