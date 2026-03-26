from pathlib import Path
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
import glob
import config

class DocumentChunker:
    def __init__(self):
        self._parent_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=config.HEADERS_TO_SPLIT_ON,
            strip_headers=False
        )

        self._child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHILD_CHUNK_SIZE,
            chunk_overlap=config.CHILD_CHUNK_OVERLAP
        )

        self._min_parent_size = config.MIN_PARENT_SIZE
        self._max_parent_size = config.MAX_PARENT_SIZE

    def create_chunks(self, path_dir=config.MARKDOWN_DIR):
        all_parent_chunks, all_child_chunks = [], []

        for path_str in sorted(glob.glob(path_dir, "*.md")):
            path = Path(path_str)

            all_parent_chunks, all_child_chunks = self.create_chunks_single(path)
            all_parent_chunks.extend(all_parent_chunks)
            all_child_chunks.extend(all_child_chunks)
        
        return all_parent_chunks, all_child_chunks
    
    def create_chunks_single(self, path):
        with open(path, "r", encoding="utf-8") as f:
            parent_chunks = self._parent_splitter.split_text(f.read())
        
        merged_parents = self._merge_small_parents(parent_chunks)
        splited_parents = self._split_large_parents(merged_parents)
        cleaned_parents = self._clean_small_parents(splited_parents)
        all_parent_pairs, all_child_chunks = [], []
        self._create_child_chunks(all_parent_pairs, all_child_chunks, cleaned_parents, path)
        return all_parent_pairs, all_child_chunks
        
    
    def _merge_small_parents(self, chunks):
        if not chunks:
            return []
        
        merged, current = [], None

        for chunk in chunks:
            if current is None:
                current = chunk
            else:
                current.page_content += "\n\n" + chunk.page_content
                for k, v in chunk.metadata.items():
                    if k in current.metadata:
                        current.metadata[k] = f"{current.metadata[k]} -> {v}"
                    else:
                        current.metadata[k] = v
            
            if len(current.page_content) >= self._max_parent_size:
                merged.append(current)
                current = None

        if current:
            if merged:
                merged[-1].page_content += "\n\n" + current.page_content
                for k, v in current.metadata.items():
                    if k in merged[-1].metadata:
                        merged[-1].metadata[k] = f"{merged[-1].metadata[k]} -> {v}"
                    else:
                        merged[-1].metadata[k] = v
            else:
                merged.append(current)
        
        return merged
    
    def _split_large_parents(self, chunks):
        split_chunks = []

        for chunk in chunks:
            if len(chunk.page_content) <= self._max_parent_size:
                split_chunks.append(chunk)
            else:
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self._max_parent_size,
                    chunk_overlap=config.CHILD_CHUNK_OVERLAP
                )
                split_chunks.extend(splitter.split_documents([chunk]))
        
        return split_chunks
    
    def _clean_small_parents(self, chunks):
        cleaned = []

        for i, chunk in enumerate(chunks):
            if len(chunk.page_content) < self._min_parent_size:
                if cleaned:
                    cleaned[-1].page_content += "\n\n" + chunk.page_content
                    for k, v in chunk.metadata.items():
                        if k in cleaned[-1].metadata:
                            cleaned[-1].metadata[k] = f"{cleaned[-1].metadata[k]} -> {v}"
                        else:
                            cleaned[-1].metadata[k] = v
                elif i < len(chunks) - 1:
                    chunks[i + 1].page_content = chunk.page_content + "\n\n" + chunks[i + 1].page_content
                    for k, v in chunk.metadata.items():
                        if k in chunks[i + 1].metadata:
                            chunks[i + 1].metadata[k] = f"{v} -> {chunks[i + 1].metadata[k]}"
                        else:
                            chunks[i + 1].metadata[k] = v
                else:
                    cleaned.append(chunk)
            else:
                cleaned.append(chunk)
        
        return cleaned
    
    def _create_child_chunks(self, all_parent_pairs, all_child_chunks, parent_chunks, doc_path):
        for i, chunk in enumerate(parent_chunks):
            parent_id = f"{doc_path.stem}_parent_{i}"
            chunk.metadata.update({"source": str(doc_path.stem)+".pdf", "parent_id": parent_id})
            all_parent_pairs.append((parent_id, chunk))
            all_child_chunks.extend(self._child_splitter.split_documents([chunk]))    
        