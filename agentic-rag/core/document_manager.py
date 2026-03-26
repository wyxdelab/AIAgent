from pathlib import Path
import config
import shutil
from utils import pdf_to_markdown
class DocumentManager:

    def __init__(self, rag_system):
        self.rag_system = rag_system
        self.markdown_dir = Path(config.MARKDOWN_DIR)
        self.markdown_dir.mkdir(parents=True, exist_ok=True)

    def add_document(self, document_paths, progress_callback = None):
        if not document_paths:
            return 0, 0

        document_paths = [document_paths] if isinstance(document_paths, str) else document_paths
        document_paths = [p for p in document_paths if p and Path(p).suffix.lower() in [".pdf", ".md"]]

        if not document_paths:
            return 0, 0
        
        added = 0
        skipped = 0
        
        for idx, doc_path in enumerate(document_paths):
            if progress_callback:
                progress_callback((idx + 1) / len(document_paths), f"Processing {Path(doc_path).name}")
            
            doc_name = Path(doc_path).stem
            md_path = self.markdown_dir / f"{doc_name}.md"
            if md_path.exists():
                skipped += 1
                continue
            
            try:
                if Path(doc_path).suffix.lower() == '.md':
                    shutil.copy(doc_path, md_path)
                else:
                    pdf_to_markdown(doc_path, overwrite=False)
                
                parent_chunks, child_chunks = self.rag_system.document_chunker.create_chunks_single(md_path)

                if not child_chunks:
                    skipped += 1
                    continue
                
                collection = self.rag_system.vector_db_manager.get_collection(config.CHILD_COLLECTION)
                collection.add_documents(child_chunks)
                self.rag_system.parent_store_manager.save_many(parent_chunks)

                added += 1
            except Exception as e:
                print(f"Error processing {doc_path}: {e}")
                skipped += 1
        
        return added, skipped
    def get_all_documents(self):
        if not self.markdown_dir.exists():
            return []
        return sorted([p.name.replace(".md", ".pdf") for p in self.markdown_dir.glob("*.md")])
    
    def clear_all_documents(self):
        if not self.markdown_dir.exists():
            return
        shutil.rmtree(self.markdown_dir)
        self.markdown_dir.mkdir(parents=True, exist_ok=True)

        self.rag_system.vector_db_manager.delete_collection(config.CHILD_COLLECTION)
        self.rag_system.vector_db_manager.create_collection(config.CHILD_COLLECTION)
        self.rag_system.parent_store_manager.clear_store()