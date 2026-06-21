#include "bm25_engine.h"
#include "tokenizer.h"
#include "inverted_index.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(_cpp_bm25, m) {
	m.doc() = "C++ BM25Okapi engine with pybind11 bindings";

    // ----- Posting struct -----
	py::class_<cpp_bm25::Posting>(m, "Posting")
		.def_readonly("doc_id", &cpp_bm25::Posting::doc_id)
		.def_readonly("term_freq", &cpp_bm25::Posting::term_freq)
		.def_readonly("positions", &cpp_bm25::Posting::positions);

    // ----- Tokenizer class -----
	py::class_<cpp_bm25::Tokenizer>(m, "Tokenizer")
		.def(py::init<bool>(),
			py::arg("remove_stop_words") = true,
			"Construct a Tokenizer. Stop-word removal is enabled by default.")
        .def("tokenize",
            &cpp_bm25::Tokenizer::Tokenize,
            py::arg("text"),
            "Tokenize the given text into a list of lowercase tokens.")
        .def("add_stop_word",
            &cpp_bm25::Tokenizer::AddStopWord,
            py::arg("word"),
            "Add a custom stop word.")
        .def("remove_stop_word",
            &cpp_bm25::Tokenizer::RemoveStopWord,
            py::arg("word"),
            "Remove a stop word.")
        .def("get_stop_words",
            &cpp_bm25::Tokenizer::GetStopWords,
            "Return the current set of stop words.");

    // ----- BM25Engine class -----
    py::class_<cpp_bm25::BM25Engine>(m, "BM25Engine")
        .def(py::init<>(),
            "Construct a BM25 engine with default parameters (k1=1.5, b=0.75).")
        .def(py::init<double, double>(),
            py::arg("k1"),
            py::arg("b"),
            "Construct a BM25 engine with custom k1 and b parameters.")
        .def("add_document",
            &cpp_bm25::BM25Engine::AddDocument,
            py::arg("doc_id"),
            py::arg("text"),
            "Add a single document to the engine.")
        .def("add_documents",
            &cpp_bm25::BM25Engine::AddDocuments,
            py::arg("doc_ids"),
            py::arg("texts"),
            "Add multiple documents at once.")
        .def("query",
            &cpp_bm25::BM25Engine::Query,
            py::arg("query_text"),
            py::arg("top_k") = 10,
            "Query the engine and return the top_k results as "
            "a list of (doc_id, score) tuples sorted by descending score.")
        .def("get_document_count",
            &cpp_bm25::BM25Engine::GetDocumentCount,
            "Return the number of documents in the index.")
        .def("clear",
            &cpp_bm25::BM25Engine::Clear,
            "Remove all documents from the engine.")
        .def("get_k1",
            &cpp_bm25::BM25Engine::GetK1,
            "Return the k1 parameter.")
        .def("get_b",
            &cpp_bm25::BM25Engine::GetB,
            "Return the b parameter.");
}