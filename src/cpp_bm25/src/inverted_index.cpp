#include "inverted_index.h"

namespace cpp_bm25 {
void InvertedIndex::AddPosting(const std::string& term, const Posting& posting) {
	std::unique_lock<std::shared_mutex> lock(mutex_);

	auto& postings = index_[term];

	// Check if a posting for this doc_id already exists in the list.
	auto it = std::find_if(postings.begin(), postings.end(),
						   [&posting](const Posting& p) {
						   	return p.doc_id == posting.doc_id;
						   });

	if (it != postings.end()) {
		// Replace the existing posting.
		*it = posting;
	} else {
		postings.push_back(posting);
	}
}

std::vector<Posting> InvertedIndex::GetPostings(const std::string& term) const {
	std::shared_lock<std::shared_mutex> lock(mutex_);

	auto it = index_.find(term);
	if (it == index_.end()) {
		return {};
	}

	return it->second;
}

int64_t InvertedIndex::GetDocumentFrequency(const std::string& term) const {
	std::shared_lock<std::shared_mutex> lock(mutex_);

	auto it = index_.find(term);
	if (it == index_.end()) {
		return 0;
	}

	return static_cast<int64_t>(it->second.size());
}

double InvertedIndex::GetAverageDocumentLength() const {
	std::shared_lock<std::shared_mutex> lock(mutex_);

	if (doc_count_ == 0) {
		return 0.0;
	}
	return static_cast<double>(total_doc_length_) /
		   static_cast<double>(doc_count_);
}

int64_t InvertedIndex::GetDocumentLength(int64_t doc_id) const {
	std::shared_lock<std::shared_mutex> lock(mutex_);

	auto it = doc_lengths_.find(doc_id);
	if (it == doc_lengths_.end()) {
		return 0;
	}
	return it->second;
}

void InvertedIndex::SetDocumentLength(int64_t doc_id, int64_t length) {
	std::unique_lock<std::shared_mutex> lock(mutex_);
	doc_lengths_[doc_id] = length;
}

void InvertedIndex::Clear() {
	std::unique_lock<std::shared_mutex> lock(mutex_);
	index_.clear();
	doc_lengths_.clear();
	doc_count_ = 0;
	total_doc_length_ = 0;
}

bool InvertedIndex::Empty() const {
	std::shared_lock<std::shared_mutex> lock(mutex_);
	return index_.empty();
}

} // namespace cpp_bm25