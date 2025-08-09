# src/index_graph/pdf_visual_extractor.py
"""
Extracteur de graphiques et tableaux depuis les PDFs pour l'indexation ANSD.
"""

import os
import csv
import io
import re
from pathlib import Path
from typing import List, Tuple, Dict, Any
import logging

# Import conditionnel pour éviter les erreurs si les bibliothèques ne sont pas installées
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    print("⚠️ PyMuPDF non installé. Installez avec: pip install PyMuPDF")

try:
    import camelot
    CAMELOT_AVAILABLE = True
except ImportError:
    CAMELOT_AVAILABLE = False
    print("⚠️ Camelot non installé. Installez avec: pip install camelot-py[cv]")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("⚠️ Pillow non installé. Installez avec: pip install Pillow")


import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class PDFVisualExtractor:
    """
    Extrait les graphiques et tableaux des PDFs et crée les index correspondants.
    """
    
    def __init__(
        self,
        output_dir: str = ".",
        images_dir: str = "images",
        tables_dir: str = "tables"
    ):
        """
        Initialise l'extracteur.
        
        Args:
            output_dir: Dossier de sortie pour les index CSV
            images_dir: Dossier pour sauvegarder les images extraites
            tables_dir: Dossier pour sauvegarder les tableaux CSV
        """
        self.output_dir = Path(output_dir)
        self.images_dir = Path(images_dir)
        self.tables_dir = Path(tables_dir)
        
        # Créer les dossiers de sortie
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        
        # Chemins des fichiers d'index
        self.charts_index_path = self.output_dir / "charts_index.csv"
        self.tables_index_path = self.output_dir / "tables_index.csv"
        
        # Logger
        self.logger = logging.getLogger(__name__)
    
    def extract_images_from_pdf(self, pdf_path: Path) -> List[Tuple[int, str, Path, str]]:
        """
        Extrait les images d'un PDF.
        
        Args:
            pdf_path: Chemin vers le PDF
            
        Returns:
            Liste de tuples (page, image_name, image_path, caption)
        """
        if not PYMUPDF_AVAILABLE or not PIL_AVAILABLE:
            self.logger.error("PyMuPDF et Pillow requis pour l'extraction d'images")
            return []
        
        images = []
        
        try:
            doc = fitz.open(str(pdf_path))
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Extraire le texte de la page pour les légendes
                page_text = page.get_text()
                captions = self._extract_captions(page_text, "Graphique")
                
                # Extraire les images
                image_list = page.get_images(full=True)
                
                for img_index, img in enumerate(image_list):
                    try:
                        # Récupérer l'image
                        xref = img[0]
                        pix = fitz.Pixmap(doc, xref)
                        
                        # Convertir en RGB si nécessaire
                        if pix.n - pix.alpha < 4:  # GRAY ou RGB
                            img_data = pix.tobytes("png")
                        else:  # CMYK
                            pix1 = fitz.Pixmap(fitz.csRGB, pix)
                            img_data = pix1.tobytes("png")
                            pix1 = None
                        
                        pix = None
                        
                        # Filtrer les petites images (probablement des icônes)
                        img_pil = Image.open(io.BytesIO(img_data))
                        if img_pil.width < 100 or img_pil.height < 100:
                            continue
                        
                        # Générer le nom du fichier
                        caption = captions[img_index] if img_index < len(captions) else ""
                        safe_caption = self._sanitize_filename(caption)
                        
                        if safe_caption:
                            filename = f"{safe_caption}_p{page_num + 1}_i{img_index + 1}.png"
                        else:
                            filename = f"{pdf_path.stem}_p{page_num + 1}_i{img_index + 1}.png"
                        
                        # Sauvegarder l'image
                        image_path = self.images_dir / filename
                        img_pil.save(image_path)
                        
                        # Générer l'ID unique
                        image_id = image_path.stem
                        
                        images.append((page_num + 1, image_id, image_path, caption))
                        
                        self.logger.info(f"Image extraite: {filename} (page {page_num + 1})")
                        
                    except Exception as e:
                        self.logger.error(f"Erreur extraction image {img_index} page {page_num + 1}: {e}")
                        continue
            
            doc.close()
            
        except Exception as e:
            self.logger.error(f"Erreur traitement PDF {pdf_path}: {e}")
        
        return images
    
    def extract_tables_from_pdf(self, pdf_path: Path) -> List[Tuple[int, int, Path, str]]:
        """
        Extrait les tableaux d'un PDF.
        
        Args:
            pdf_path: Chemin vers le PDF
            
        Returns:
            Liste de tuples (page, table_index, table_path, caption)
        """
        if not CAMELOT_AVAILABLE:
            self.logger.error("Camelot requis pour l'extraction de tableaux")
            return []
        
        tables = []
        
        try:
            # Extraire le texte pour les légendes
            if PYMUPDF_AVAILABLE:
                doc = fitz.open(str(pdf_path))
                page_texts = [page.get_text() for page in doc]
                doc.close()
            else:
                page_texts = []
            
            # Extraire les tableaux avec Camelot
            for flavor in ["lattice", "stream"]:
                try:
                    tables_found = camelot.read_pdf(
                        str(pdf_path), 
                        flavor=flavor, 
                        pages="all"
                    )
                    
                    for table_index, table in enumerate(tables_found):
                        page_num = table.page
                        
                        # Trouver la légende
                        caption = ""
                        if page_num - 1 < len(page_texts):
                            page_text = page_texts[page_num - 1]
                            captions = self._extract_captions(page_text, "Tableau")
                            if table_index < len(captions):
                                caption = captions[table_index]
                        
                        # Générer le nom du fichier
                        safe_caption = self._sanitize_filename(caption)
                        
                        if safe_caption:
                            filename = f"{safe_caption}_p{page_num}_t{table_index + 1}.csv"
                        else:
                            filename = f"{pdf_path.stem}_p{page_num}_t{table_index + 1}.csv"
                        
                        # Sauvegarder le tableau
                        table_path = self.tables_dir / filename
                        table.to_csv(str(table_path))
                        
                        # Générer l'ID unique
                        table_id = table_path.stem
                        
                        tables.append((page_num, table_index + 1, table_path, caption))
                        
                        self.logger.info(f"Tableau extrait: {filename} (page {page_num})")
                
                except Exception as e:
                    self.logger.warning(f"Erreur extraction tableaux avec {flavor}: {e}")
                    continue
        
        except Exception as e:
            self.logger.error(f"Erreur extraction tableaux de {pdf_path}: {e}")
        
        return tables
    
    def _extract_captions(self, text: str, prefix: str) -> List[str]:
        """
        Extrait les légendes commençant par un préfixe donné.
        
        Args:
            text: Texte de la page
            prefix: Préfixe à rechercher (ex: "Graphique", "Tableau")
            
        Returns:
            Liste des légendes trouvées
        """
        pattern = rf"^{prefix}\s+[\w\-\.\s]+\s*[:]\s*.+$"
        matches = re.findall(pattern, text, re.MULTILINE | re.IGNORECASE)
        return [match.strip() for match in matches]
    
    def _sanitize_filename(self, text: str, max_length: int = 50) -> str:
        """
        Nettoie un texte pour en faire un nom de fichier valide.
        
        Args:
            text: Texte à nettoyer
            max_length: Longueur maximale
            
        Returns:
            Nom de fichier nettoyé
        """
        if not text:
            return ""
        
        # Supprimer les caractères spéciaux
        cleaned = re.sub(r'[^\w\s\-]', '', text)
        # Remplacer les espaces par des underscores
        cleaned = re.sub(r'\s+', '_', cleaned)
        # Limiter la longueur
        cleaned = cleaned[:max_length]
        # Supprimer les underscores en début/fin
        cleaned = cleaned.strip('_')
        
        return cleaned
    
    def process_pdf(self, pdf_path: Path) -> Dict[str, Any]:
        """
        Traite un PDF complet (images + tableaux).
        
        Args:
            pdf_path: Chemin vers le PDF
            
        Returns:
            Statistiques d'extraction
        """
        self.logger.info(f"Traitement du PDF: {pdf_path.name}")
        
        stats = {
            "pdf_path": str(pdf_path),
            "images_extracted": 0,
            "tables_extracted": 0,
            "errors": []
        }
        
        try:
            # Extraire les images
            images = self.extract_images_from_pdf(pdf_path)
            stats["images_extracted"] = len(images)
            
            # Mettre à jour l'index des graphiques
            self._update_charts_index(images, pdf_path)
            
        except Exception as e:
            error_msg = f"Erreur extraction images: {e}"
            stats["errors"].append(error_msg)
            self.logger.error(error_msg)
        
        try:
            # Extraire les tableaux
            tables = self.extract_tables_from_pdf(pdf_path)
            stats["tables_extracted"] = len(tables)
            
            # Mettre à jour l'index des tableaux
            self._update_tables_index(tables, pdf_path)
            
        except Exception as e:
            error_msg = f"Erreur extraction tableaux: {e}"
            stats["errors"].append(error_msg)
            self.logger.error(error_msg)
        
        return stats
    
    def _update_charts_index(self, images: List[Tuple], pdf_path: Path):
        """Met à jour le fichier d'index des graphiques."""
        
        # Créer le fichier avec en-têtes s'il n'existe pas
        if not self.charts_index_path.exists():
            with open(self.charts_index_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["image_id", "pdf_path", "page", "image_path", "caption"])
        
        # Ajouter les nouvelles entrées
        with open(self.charts_index_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for page, image_id, image_path, caption in images:
                writer.writerow([
                    image_id,
                    str(pdf_path),
                    page,
                    str(image_path),
                    caption
                ])
    
    def _update_tables_index(self, tables: List[Tuple], pdf_path: Path):
        """Met à jour le fichier d'index des tableaux."""
        
        # Créer le fichier avec en-têtes s'il n'existe pas
        if not self.tables_index_path.exists():
            with open(self.tables_index_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["table_id", "pdf_path", "page", "table_path", "caption"])
        
        # Ajouter les nouvelles entrées
        with open(self.tables_index_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for page, table_index, table_path, caption in tables:
                writer.writerow([
                    table_path.stem,
                    str(pdf_path),
                    page,
                    str(table_path),
                    caption
                ])
    
    def process_directory(self, pdf_directory: Path) -> Dict[str, Any]:
        """
        Traite tous les PDFs d'un dossier.
        
        Args:
            pdf_directory: Dossier contenant les PDFs
            
        Returns:
            Statistiques globales d'extraction
        """
        if not pdf_directory.exists():
            raise FileNotFoundError(f"Dossier non trouvé: {pdf_directory}")
        
        # Supprimer les anciens index pour recommencer
        if self.charts_index_path.exists():
            self.charts_index_path.unlink()
        if self.tables_index_path.exists():
            self.tables_index_path.unlink()
        
        pdf_files = list(pdf_directory.glob("**/*.pdf"))
        
        global_stats = {
            "total_pdfs": len(pdf_files),
            "processed_pdfs": 0,
            "total_images": 0,
            "total_tables": 0,
            "failed_pdfs": [],
            "processing_errors": []
        }
        
        self.logger.info(f"Début extraction de {len(pdf_files)} PDFs depuis {pdf_directory}")
        
        for i, pdf_path in enumerate(pdf_files, 1):
            self.logger.info(f"Traitement {i}/{len(pdf_files)}: {pdf_path.name}")
            
            try:
                stats = self.process_pdf(pdf_path)
                
                global_stats["processed_pdfs"] += 1
                global_stats["total_images"] += stats["images_extracted"]
                global_stats["total_tables"] += stats["tables_extracted"]
                
                if stats["errors"]:
                    global_stats["processing_errors"].extend(stats["errors"])
                
                self.logger.info(
                    f"  ✅ {stats['images_extracted']} images, "
                    f"{stats['tables_extracted']} tableaux extraits"
                )
                
            except Exception as e:
                error_msg = f"Échec traitement {pdf_path.name}: {e}"
                global_stats["failed_pdfs"].append(pdf_path.name)
                global_stats["processing_errors"].append(error_msg)
                self.logger.error(error_msg)
        
        # Rapport final
        self.logger.info("="*60)
        self.logger.info("RAPPORT D'EXTRACTION VISUELLE")
        self.logger.info("="*60)
        self.logger.info(f"PDFs traités: {global_stats['processed_pdfs']}/{global_stats['total_pdfs']}")
        self.logger.info(f"Images extraites: {global_stats['total_images']}")
        self.logger.info(f"Tableaux extraits: {global_stats['total_tables']}")
        
        if global_stats["failed_pdfs"]:
            self.logger.warning(f"PDFs échoués: {', '.join(global_stats['failed_pdfs'])}")
        
        if global_stats["processing_errors"]:
            self.logger.warning(f"Erreurs de traitement: {len(global_stats['processing_errors'])}")
        
        self.logger.info(f"Index créés:")
        self.logger.info(f"  📊 Graphiques: {self.charts_index_path}")
        self.logger.info(f"  📋 Tableaux: {self.tables_index_path}")
        
        return global_stats


def extract_visual_content_from_pdfs(
    pdf_directory: str,
    output_dir: str = ".",
    images_dir: str = "images", 
    tables_dir: str = "tables"
) -> Dict[str, Any]:
    """
    Fonction utilitaire pour extraire le contenu visuel des PDFs.
    
    Args:
        pdf_directory: Dossier contenant les PDFs
        output_dir: Dossier de sortie pour les index
        images_dir: Dossier pour les images
        tables_dir: Dossier pour les tableaux
        
    Returns:
        Statistiques d'extraction
    """
    # Configuration du logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Créer l'extracteur
    extractor = PDFVisualExtractor(
        output_dir=output_dir,
        images_dir=images_dir,
        tables_dir=tables_dir
    )
    
    # Traiter le dossier
    return extractor.process_directory(Path(pdf_directory))


if __name__ == "__main__":
    """Script principal pour tester l'extraction."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Extraction de contenu visuel depuis PDFs")
    parser.add_argument("pdf_dir", help="Dossier contenant les PDFs")
    parser.add_argument("--output", default=".", help="Dossier de sortie")
    parser.add_argument("--images", default="images", help="Dossier pour images")
    parser.add_argument("--tables", default="tables", help="Dossier pour tableaux")
    
    args = parser.parse_args()
    
    print("🚀 EXTRACTION DE CONTENU VISUEL DEPUIS PDFS")
    print("="*50)
    
    try:
        stats = extract_visual_content_from_pdfs(
            pdf_directory=args.pdf_dir,
            output_dir=args.output,
            images_dir=args.images,
            tables_dir=args.tables
        )
        
        print(f"\n✅ Extraction terminée!")
        print(f"📊 {stats['total_images']} graphiques extraits")
        print(f"📋 {stats['total_tables']} tableaux extraits")
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        sys.exit(1)