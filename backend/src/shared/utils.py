"""Shared utility functions used in the project.

Functions:
    format_docs: Convert documents to an xml-formatted string.
    load_chat_model: Load a chat model from a model name.
"""
import os
from typing import Optional, List, Dict

from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from pinecone import Index, Pinecone, ServerlessSpec
from urllib.parse import urljoin 
import re
from typing import List, Dict, Any
from datetime import datetime
import csv
from pathlib import Path
import fitz  # PyMuPDF
import unicodedata

def _format_doc(doc: Document) -> str:
    """Format a single document as XML.

    Args:
        doc (Document): The document to format.

    Returns:
        str: The formatted document as an XML string.
    """
    metadata = doc.metadata or {}
    meta = "".join(f" {k}={v!r}" for k, v in metadata.items())
    if meta:
        meta = f" {meta}"

    return f"<document{meta}>\n{doc.page_content}\n</document>"


def format_docs(docs: Optional[list[Document]]) -> str:
    """Format a list of documents as XML.

    This function takes a list of Document objects and formats them into a single XML string.

    Args:
        docs (Optional[list[Document]]): A list of Document objects to format, or None.

    Returns:
        str: A string containing the formatted documents in XML format.

    Examples:
        >>> docs = [Document(page_content="Hello"), Document(page_content="World")]
        >>> print(format_docs(docs))
        <documents>
        <document>
        Hello
        </document>
        <document>
        World
        </document>
        </documents>

        >>> print(format_docs(None))
        <documents></documents>
    """
    if not docs:
        return "<documents></documents>"
    formatted = "\n".join(_format_doc(doc) for doc in docs)
    return f"""<documents>
{formatted}
</documents>"""

def load_pinecone_index(index_name: str):
    """
    Charge un index Pinecone existant, ou le crée automatiquement s’il n’existe pas.
    """
    pinecone_client = Pinecone(
        api_key=os.environ["PINECONE_API_KEY"],
        environment=os.environ["PINECONE_ENVIRONMENT"]
    )

    indexes = pinecone_client.list_indexes().names()
    print("🔎 Index disponibles :", indexes)

    if index_name not in indexes:
        print(f"⚠️ L'index '{index_name}' n'existe pas. Création...")
        pinecone_client.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec= ServerlessSpec(
                cloud="aws",  # or "gcp"
                region="us-east-1"
            )
        )
        print(f"✅ Index '{index_name}' créé.")

    return pinecone_client.Index(index_name)

def load_chat_model(fully_specified_name: str) -> BaseChatModel:
    """Load a chat model from a fully specified name.

    Args:
        fully_specified_name (str): String in the format 'provider/model'.
    """
    if "/" in fully_specified_name:
        provider, model = fully_specified_name.split("/", maxsplit=1)
    else:
        provider = ""
        model = fully_specified_name
    return init_chat_model(model, model_provider=provider)
# =============================================================================
# FICHIER 4: src/shared/utils.py
# =============================================================================
# AJOUTEZ CES FONCTIONS À LA FIN DE VOTRE FICHIER EXISTANT (ne pas remplacer le contenu existant)

# =============================================================================
# FONCTIONS UTILITAIRES AMÉLIORÉES POUR L'ANSD
# =============================================================================

import re
from typing import List, Dict, Any
from datetime import datetime

def detect_ansd_survey_type(text: str) -> List[str]:
    """Détecte le type d'enquête ANSD dans un texte."""
    
    text_lower = text.lower()
    surveys = []
    
    # Dictionnaire des enquêtes avec leurs variantes
    survey_patterns = {
        "rgph": ["rgph", "recensement", "population", "habitat", "recensement général"],
        "eds": ["eds", "enquête démographique", "santé", "demographic health survey"],
        "esps": ["esps", "pauvreté", "enquête de suivi", "poverty"],
        "ehcvm": ["ehcvm", "conditions de vie", "ménages", "budget", "consommation"],
        "enes": ["enes", "emploi", "chômage", "activité économique"],
        "comptes_nationaux": ["pib", "comptes nationaux", "économie", "croissance"],
    }
    
    for survey, patterns in survey_patterns.items():
        if any(pattern in text_lower for pattern in patterns):
            surveys.append(survey)
    
    return surveys

def extract_numerical_data(text: str) -> List[Dict[str, Any]]:
    """Extrait les données numériques d'un texte avec leur contexte."""
    
    numerical_data = []
    
    # Patterns pour différents types de données
    patterns = [
        # Pourcentages
        (r'(\d+(?:[.,]\d+)?)\s*%', 'percentage'),
        # Nombres avec unités monétaires
        (r'(\d+(?:[.,]\d+)?)\s*(fcfa|francs?|f\s*cfa)', 'currency'),
        # Populations en millions/milliards
        (r'(\d+(?:[.,]\d+)?)\s*(millions?|milliards?)', 'population'),
        # Années
        (r'\b(20\d{2})\b', 'year'),
        # Nombres simples avec contexte
        (r'(\d+(?:[.,]\d+)?)', 'number'),
    ]
    
    for pattern, data_type in patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            # Extraire le contexte autour du nombre (50 caractères avant et après)
            start_context = max(0, match.start() - 50)
            end_context = min(len(text), match.end() + 50)
            context = text[start_context:end_context].strip()
            
            numerical_data.append({
                'value': match.group(1),
                'type': data_type,
                'context': context,
                'position': match.span()
            })
    
    return numerical_data

def format_ansd_metadata(metadata: Dict[str, Any]) -> str:
    """Formate les métadonnées spécifiques aux documents ANSD."""
    
    formatted_parts = []
    
    # Source principale
    if 'source' in metadata:
        formatted_parts.append(f"📄 Source: {metadata['source']}")
    
    # Nom du document
    if 'pdf_name' in metadata:
        formatted_parts.append(f"📋 Document: {metadata['pdf_name']}")
    
    # Numéro de page
    if 'page_num' in metadata:
        formatted_parts.append(f"📖 Page: {metadata['page_num']}")
    
    # Date d'indexation
    if 'indexed_at' in metadata:
        date_str = metadata['indexed_at'][:10] if len(metadata['indexed_at']) >= 10 else metadata['indexed_at']
        formatted_parts.append(f"🕐 Indexé: {date_str}")
    
    # Type de document
    if 'type' in metadata:
        formatted_parts.append(f"📊 Type: {metadata['type']}")
    
    # Taille du fichier (si disponible)
    if 'file_size' in metadata:
        size_mb = metadata['file_size'] / (1024 * 1024)
        formatted_parts.append(f"💾 Taille: {size_mb:.1f} MB")
    
    return " | ".join(formatted_parts) if formatted_parts else "📄 Métadonnées non disponibles"

def calculate_document_relevance_score(doc_content: str, query: str, survey_weights: Dict[str, float] = None) -> float:
    """Calcule un score de pertinence pour un document par rapport à une requête."""
    
    if not doc_content or not query:
        return 0.0
    
    doc_lower = doc_content.lower()
    query_lower = query.lower()
    query_words = set(query_lower.split())
    
    score = 0.0
    
    # Score basé sur les mots-clés de la requête
    for word in query_words:
        if len(word) > 3:  # Ignorer les mots très courts
            word_count = doc_lower.count(word)
            score += word_count * 2
    
    # Bonus pour les termes ANSD spécifiques
    ansd_terms = {
        'rgph': 5,
        'eds': 5,
        'esps': 5,
        'ehcvm': 5,
        'enes': 5,
        'ansd': 3,
        'sénégal': 3,
        'recensement': 4,
        'enquête': 3,
        'statistique': 3
    }
    
    for term, weight in ansd_terms.items():
        if term in doc_lower:
            score += weight
    
    # Bonus pour les données numériques
    numerical_patterns = [
        r'\d+[.,]\d+\s*%',  # Pourcentages
        r'\d+\s*(millions?|milliards?)',  # Grandes populations
        r'\d{4}',  # Années
    ]
    
    for pattern in numerical_patterns:
        matches = len(re.findall(pattern, doc_lower))
        score += matches * 2
    
    # Appliquer les poids des enquêtes si fournis
    if survey_weights:
        surveys = detect_ansd_survey_type(doc_content)
        for survey in surveys:
            if survey in survey_weights:
                score *= survey_weights[survey]
    
    return score

def clean_ansd_text(text: str) -> str:
    """Nettoie le texte des documents ANSD pour améliorer la lisibilité."""
    
    if not text:
        return ""
    
    # Supprimer les caractères de contrôle
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    
    # Normaliser les espaces
    text = re.sub(r'\s+', ' ', text)
    
    # Corriger les problèmes d'encodage courants
    replacements = {
        'Ã ': 'à ',
        'Ã©': 'é',
        'Ã¨': 'è',
        'Ã§': 'ç',
        'Ã´': 'ô',
        'Ã¢': 'â',
        'Ã®': 'î',
        'Ã»': 'û',
        'Ã¹': 'ù',
    }
    
    for wrong, correct in replacements.items():
        text = text.replace(wrong, correct)
    
    return text.strip()

def extract_ansd_indicators(text: str) -> List[Dict[str, str]]:
    """Extrait les indicateurs statistiques spécifiques à l'ANSD."""
    
    indicators = []
    
    # Patterns pour différents types d'indicateurs
    indicator_patterns = [
        # Taux démographiques
        (r'taux\s+(?:de\s+)?(?:natalité|mortalité|fécondité|croissance)\s*:\s*([^.\n]+)', 'démographique'),
        # Indicateurs économiques
        (r'(?:pib|produit\s+intérieur\s+brut)\s*:\s*([^.\n]+)', 'économique'),
        # Indicateurs de pauvreté
        (r'(?:taux\s+de\s+pauvreté|incidence\s+de\s+la\s+pauvreté)\s*:\s*([^.\n]+)', 'pauvreté'),
        # Indicateurs d'éducation
        (r'taux\s+(?:d\s*\'?\s*alphabétisation|de\s+scolarisation)\s*:\s*([^.\n]+)', 'éducation'),
        # Indicateurs de santé
        (r'(?:espérance\s+de\s+vie|mortalité\s+(?:infantile|maternelle))\s*:\s*([^.\n]+)', 'santé'),
    ]
    
    for pattern, category in indicator_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            indicators.append({
                'category': category,
                'indicator': match.group(0),
                'value': match.group(1).strip(),
                'position': match.span()
            })
    
    return indicators

def validate_ansd_response(response: str) -> Dict[str, Any]:
    """Valide qu'une réponse contient les éléments requis pour l'ANSD."""
    
    validation = {
        'has_numerical_data': False,
        'has_source_citation': False,
        'has_year_reference': False,
        'has_ansd_terminology': False,
        'quality_score': 0.0,
        'suggestions': []
    }
    
    response_lower = response.lower()
    
    # Vérifier la présence de données numériques
    if re.search(r'\d+(?:[.,]\d+)?(?:\s*%|\s*millions?|\s*milliards?)', response):
        validation['has_numerical_data'] = True
        validation['quality_score'] += 25
    else:
        validation['suggestions'].append("Ajouter des données chiffrées précises")
    
    # Vérifier les citations de sources
    if any(term in response_lower for term in ['source:', 'rgph', 'eds', 'esps', 'ehcvm', 'enes', 'ansd']):
        validation['has_source_citation'] = True
        validation['quality_score'] += 25
    else:
        validation['suggestions'].append("Citer les sources ANSD spécifiques")
    
    # Vérifier les références temporelles
    if re.search(r'20\d{2}|année\s+de\s+référence', response):
        validation['has_year_reference'] = True
        validation['quality_score'] += 25
    else:
        validation['suggestions'].append("Préciser l'année de référence des données")
    
    # Vérifier la terminologie ANSD
    ansd_terms = ['enquête', 'recensement', 'indicateur', 'statistique', 'démographique', 'sénégal']
    if any(term in response_lower for term in ansd_terms):
        validation['has_ansd_terminology'] = True
        validation['quality_score'] += 25
    else:
        validation['suggestions'].append("Utiliser la terminologie statistique appropriée")
    
    return validation

# =============================================================================
# FONCTIONS DE LOGGING SPÉCIALISÉES
# =============================================================================

def log_ansd_retrieval(query: str, documents: List, processing_time: float = None):
    """Log spécialisé pour le processus de récupération ANSD."""
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print(f"\n{'='*60}")
    print(f"🕐 ANSD RETRIEVAL LOG - {timestamp}")
    print(f"{'='*60}")
    print(f"📝 Requête: {query}")
    print(f"📚 Documents trouvés: {len(documents)}")
    
    if processing_time:
        print(f"⏱️  Temps de traitement: {processing_time:.2f}s")
    
    # Analyser les types d'enquêtes trouvées
    survey_counts = {}
    for doc in documents:
        surveys = detect_ansd_survey_type(doc.page_content)
        for survey in surveys:
            survey_counts[survey] = survey_counts.get(survey, 0) + 1
    
    if survey_counts:
        print(f"📊 Types d'enquêtes identifiées:")
        for survey, count in survey_counts.items():
            print(f"   • {survey.upper()}: {count} document(s)")
    
    print(f"{'='*60}\n")

def log_ansd_generation(response: str, validation_result: Dict = None):
    """Log spécialisé pour le processus de génération ANSD."""
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print(f"\n{'='*60}")
    print(f"🤖 ANSD GENERATION LOG - {timestamp}")
    print(f"{'='*60}")
    print(f"📏 Longueur de la réponse: {len(response)} caractères")
    
    if validation_result:
        print(f"⭐ Score de qualité: {validation_result['quality_score']}/100")
        if validation_result['suggestions']:
            print(f"💡 Suggestions d'amélioration:")
            for suggestion in validation_result['suggestions']:
                print(f"   • {suggestion}")
    
    # Extraire les données numériques mentionnées
    numerical_data = extract_numerical_data(response)
    if numerical_data:
        print(f"🔢 Données numériques détectées: {len(numerical_data)}")
        for data in numerical_data[:3]:  # Afficher les 3 premières
            print(f"   • {data['value']} ({data['type']})")
    
    print(f"{'='*60}\n")



# Chemin fixes à la racine du projet
IMAGES_DIR = Path("./images")
INDEX_CSV = Path("./charts_index.csv")


def generate_charts_index(pdf_root: Path) -> None:
    """
    Extrait toutes les images (charts) des PDF sous pdf_root,
    les sauvegarde dans IMAGES_DIR
    et reconstruit INDEX_CSV.

    :param pdf_root: Dossier racine contenant les PDF à analyser.
    """
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    with INDEX_CSV.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["image_id", "pdf_path", "page", "image_path"])

        for pdf_path in pdf_root.rglob("*.pdf"):
            doc = fitz.open(str(pdf_path))
            for page_num in range(len(doc)):
                page = doc[page_num]
                for img_index, img in enumerate(page.get_images(full=True), start=1):
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)
                    if pix.n > 4:
                        pix = fitz.Pixmap(fitz.csRGB, pix)

                    image_name = f"{pdf_path.stem}_p{page_num+1}_i{img_index}.png"
                    image_path = IMAGES_DIR / image_name
                    pix.save(str(image_path))
                    pix = None

                    image_id = image_name.replace(".png", "")
                    writer.writerow([
                        image_id,
                        str(pdf_path),
                        page_num + 1,
                        str(image_path)
                    ])

    print(f"[✔] Generated {INDEX_CSV} with images from {pdf_root}")



def sanitize_pinecone_id(text: str) -> str:
    """
    Convertit un texte en ID valide pour Pinecone (ASCII uniquement).
    
    Résout les erreurs comme:
    - 'visual_table_Tableau_VII-7_Proportion_de_célibataires_définitif_p5_t7'
    - 'visual_chart_Graphique_IV-16_Pourcentage_de_la_déclaration_de_p_p38_i2'
    
    Args:
        text: Texte source pouvant contenir des caractères non-ASCII
        
    Returns:
        ID valide pour Pinecone (ASCII uniquement)
        
    Examples:
        >>> sanitize_pinecone_id("visual_table_Tableau_VII-7_Proportion_de_célibataires_définitif_p5_t7")
        'visual_table_Tableau_VII_7_Proportion_de_celibataires_definitif_p5_t7'
        
        >>> sanitize_pinecone_id("visual_chart_Graphique_IV-16_Pourcentage_de_la_déclaration_de_p_p38_i2")
        'visual_chart_Graphique_IV_16_Pourcentage_de_la_declaration_de_p_p38_i2'
    """
    if not text:
        return f"item_{hash('empty') % 1000000}"
    
    # 1. Normalisation Unicode (é → e, à → a, ç → c, etc.)
    text = unicodedata.normalize('NFD', text)
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
    
    # 2. Remplacer tous les caractères non-ASCII par underscore
    text = re.sub(r'[^a-zA-Z0-9_-]', '_', text)
    
    # 3. Nettoyer les underscores multiples
    text = re.sub(r'_+', '_', text)
    
    # 4. Supprimer les underscores en début/fin
    text = text.strip('_')
    
    # 5. Limiter la longueur (Pinecone max 512, on prend 80 pour sécurité)
    if len(text) > 80:
        text = text[:80].rstrip('_')
    
    # 6. S'assurer qu'on a quelque chose de valide
    if not text or not text.replace('_', '').replace('-', ''):
        text = f"item_{hash(str(text)) % 1000000}"
    
    return text



def test_sanitize_pinecone_id():
    """Teste la fonction de sanitisation avec des cas réels."""
    
    test_cases = [
        # Cas problématiques réels de votre système
        "visual_table_Tableau_VII-7_Proportion_de_célibataires_définitif_p5_t7",
        "visual_chart_Graphique_IV-16_Pourcentage_de_la_déclaration_de_p_p38_i2",
        "visual_chart_Graphique_I-10_Évolution_du_rapport_de_masculinité_p27_i1",
        
        # Autres cas à tester
        "Chapitre 1- ETAT-STRUCTURE-POPULATION-Rapport-Provisoire-RGPH5_juillet2024_0_p28_i1",
        "Graphique_I-11_Pyramide_des_âges_de_la_population_p29_i1",
        "Chapitre 4 - FECONDITE-NATALITE-Rapport-Provisoire-RGPH5_juillet2024_0_p25_i2",
        
        # Cas limites
        "",
        "___---___",
        "a" * 200,  # Trop long
    ]
    
    print("🧪 TEST DE SANITISATION PINECONE IDS")
    print("=" * 80)
    
    for i, test_id in enumerate(test_cases, 1):
        fixed_id = sanitize_pinecone_id(test_id)
        
        print(f"\nTest {i}:")
        print(f"  Avant  : {test_id}")
        print(f"  Après  : {fixed_id}")
        print(f"  ASCII  : {fixed_id.isascii()}")
        print(f"  Longueur: {len(fixed_id)}")
        print(f"  Valide : {'✅' if fixed_id.isascii() and len(fixed_id) <= 80 and fixed_id else '❌'}")


# =============================================================================
# UTILITAIRES POUR DEBUGGING DE L'INDEXATION
# =============================================================================

def validate_pinecone_id(text: str) -> Dict[str, Any]:
    """
    Valide qu'un ID est compatible avec Pinecone.
    
    Args:
        text: ID à valider
        
    Returns:
        Dictionnaire avec les résultats de validation
    """
    validation = {
        'is_valid': True,
        'is_ascii': text.isascii(),
        'length_ok': len(text) <= 512,
        'not_empty': bool(text.strip()),
        'errors': [],
        'warnings': []
    }
    
    if not validation['is_ascii']:
        validation['is_valid'] = False
        validation['errors'].append(f"Contient des caractères non-ASCII: {[c for c in text if not c.isascii()]}")
    
    if not validation['length_ok']:
        validation['is_valid'] = False
        validation['errors'].append(f"Trop long: {len(text)} caractères (max 512)")
    
    if not validation['not_empty']:
        validation['is_valid'] = False
        validation['errors'].append("ID vide")
    
    if len(text) > 80:
        validation['warnings'].append(f"ID assez long: {len(text)} caractères")
    
    return validation


def diagnose_indexation_errors(error_message: str) -> Dict[str, Any]:
    """
    Diagnostique les erreurs d'indexation courantes.
    
    Args:
        error_message: Message d'erreur reçu
        
    Returns:
        Diagnostic avec solutions suggérées
    """
    diagnosis = {
        'error_type': 'unknown',
        'cause': 'Non identifiée',
        'solution': 'Vérifier les logs détaillés',
        'code_fix': None
    }
    
    error_lower = error_message.lower()
    
    # Erreur ASCII Pinecone
    if "vector id must be ascii" in error_lower:
        diagnosis.update({
            'error_type': 'pinecone_ascii',
            'cause': 'ID contient des caractères non-ASCII (accents, caractères spéciaux)',
            'solution': 'Utiliser sanitize_pinecone_id() pour nettoyer les IDs',
            'code_fix': '''
# Dans src/index_graph/graph.py, remplacer:
doc_id = f"visual_chart_{row.get('image_id', idx)}"

# Par:
from shared.utils import sanitize_pinecone_id
raw_id = f"visual_chart_{row.get('image_id', idx)}"
doc_id = sanitize_pinecone_id(raw_id)
'''
        })
    
    # Erreur de limite de taux API
    elif "rate limit" in error_lower or "429" in error_lower:
        diagnosis.update({
            'error_type': 'api_rate_limit',
            'cause': 'Trop d\'appels API simultanés vers OpenAI',
            'solution': 'Réduire visual_batch_size et ajouter des délais',
            'code_fix': '''
# Dans la configuration, réduire:
"visual_batch_size": 2  # Au lieu de 5

# Ajouter des délais entre batches:
await asyncio.sleep(2)  # 2 secondes
'''
        })
    
    # Erreur Pinecone de dimension
    elif "dimension" in error_lower:
        diagnosis.update({
            'error_type': 'pinecone_dimension',
            'cause': 'Dimension des vecteurs incorrecte',
            'solution': 'Vérifier la dimension de l\'embedding model (1536 pour text-embedding-3-small)',
        })
    
    return diagnosis


# =============================================================================
# FONCTIONS DE MONITORING POUR L'INDEXATION
# =============================================================================

def log_indexation_progress(current: int, total: int, item_type: str = "élément") -> None:
    """
    Affiche le progrès de l'indexation de manière formatée.
    
    Args:
        current: Nombre d'éléments traités
        total: Nombre total d'éléments
        item_type: Type d'élément (graphique, tableau, etc.)
    """
    percentage = (current / total) * 100 if total > 0 else 0
    bar_length = 30
    filled_length = int(bar_length * current // total) if total > 0 else 0
    bar = '█' * filled_length + '░' * (bar_length - filled_length)
    
    print(f"\r🔄 Indexation {item_type}: [{bar}] {current}/{total} ({percentage:.1f}%)", end='', flush=True)
    
    if current == total:
        print()  # Nouvelle ligne à la fin


def create_indexation_report(stats: Dict[str, Any]) -> str:
    """
    Crée un rapport détaillé de l'indexation.
    
    Args:
        stats: Statistiques d'indexation
        
    Returns:
        Rapport formaté
    """
    report_lines = [
        "📊 RAPPORT D'INDEXATION ANSD",
        "=" * 50,
        ""
    ]
    
    # Statistiques générales
    if 'total_text_chunks' in stats:
        report_lines.append(f"📝 Chunks de texte indexés: {stats['total_text_chunks']:,}")
    
    # Statistiques visuelles
    visual_stats = stats.get('visual_indexing_stats', {})
    if visual_stats:
        report_lines.extend([
            f"📊 Graphiques indexés: {visual_stats.get('charts_indexed', 0):,}",
            f"📋 Tableaux indexés: {visual_stats.get('tables_indexed', 0):,}",
            f"❌ Graphiques échoués: {visual_stats.get('charts_failed', 0):,}",
            f"❌ Tableaux échoués: {visual_stats.get('tables_failed', 0):,}",
        ])
    
    # Total
    total_visual = visual_stats.get('charts_indexed', 0) + visual_stats.get('tables_indexed', 0)
    total_content = stats.get('total_text_chunks', 0) + total_visual
    
    report_lines.extend([
        "",
        f"🎯 TOTAL INDEXÉ: {total_content:,} éléments",
        f"   ├─ Textuel: {stats.get('total_text_chunks', 0):,}",
        f"   └─ Visuel: {total_visual:,}",
        ""
    ])
    
    # Fichiers traités
    if 'processed_files' in stats:
        report_lines.append(f"✅ PDFs traités: {len(stats['processed_files'])}")
    
    if 'failed_files' in stats and stats['failed_files']:
        report_lines.append(f"❌ PDFs échoués: {len(stats['failed_files'])}")
    
    # Recommandations
    report_lines.extend([
        "",
        "💡 RECOMMANDATIONS:",
    ])
    
    if visual_stats.get('charts_failed', 0) > 0 or visual_stats.get('tables_failed', 0) > 0:
        report_lines.append("   • Vérifier les erreurs ASCII dans les logs")
        report_lines.append("   • Considérer réduire visual_batch_size")
    
    if total_content > 10000:
        report_lines.append("   • Excellent volume de données indexées!")
    elif total_content > 1000:
        report_lines.append("   • Bon volume de données indexées")
    else:
        report_lines.append("   • Vérifier si tous les PDFs ont été traités")
    
    return "\n".join(report_lines)


# =============================================================================
# FONCTION DE TEST PRINCIPALE
# =============================================================================

def run_ascii_fix_tests():
    """Lance tous les tests de correction ASCII."""
    
    print("🛠️ TESTS DE CORRECTION ASCII POUR SUN-STATS")
    print("=" * 60)
    
    # Test de la fonction principale
    test_sanitize_pinecone_id()
    
    # Test de validation
    print("\n🔍 TESTS DE VALIDATION:")
    print("-" * 30)
    
    test_ids = [
        "visual_table_Tableau_VII-7_Proportion_de_célibataires_définitif_p5_t7",  # Problématique
        "visual_chart_valid_ascii_id",  # Valide
        "a" * 600,  # Trop long
        ""  # Vide
    ]
    
    for test_id in test_ids:
        validation = validate_pinecone_id(test_id)
        status = "✅ VALIDE" if validation['is_valid'] else "❌ INVALIDE"
        print(f"{status}: {test_id[:50]}{'...' if len(test_id) > 50 else ''}")
        
        if validation['errors']:
            for error in validation['errors']:
                print(f"    ❌ {error}")
        
        if validation['warnings']:
            for warning in validation['warnings']:
                print(f"    ⚠️ {warning}")
    
    print("\n✅ Tests terminés. Utilisez sanitize_pinecone_id() dans votre code d'indexation.")


if __name__ == "__main__":
    # Pour tester directement ce fichier
    run_ascii_fix_tests()