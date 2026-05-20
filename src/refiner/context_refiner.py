def refine_context(docs):

    seen = set()

    refined_chunks = []

    for doc in docs:

        content = (
            doc.page_content.strip()
        )

        # Remove duplicates
        if content not in seen:

            seen.add(content)

            # Reduce huge chunks
            refined_chunks.append(
                content[:800]
            )

    refined_context = "\n\n".join(
        refined_chunks
    )

    return refined_context