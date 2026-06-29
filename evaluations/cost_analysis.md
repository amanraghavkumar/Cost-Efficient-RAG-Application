# Cost Analysis

This project uses ChromaDB as the vector database because it can run locally without any additional database subscription cost. The main objective of this project is to evaluate whether a local vector database can be used as a low-cost alternative to a managed vector database.

## Cost Comparison

| Number of Vectors | ChromaDB (Local)                    | Managed Vector Database   |
| ----------------- | ----------------------------------- | ------------------------- |
| 100K vectors      | No additional database cost         | Additional monthly cost   |
| 1M vectors        | May require more storage and memory | Higher monthly cost       |
| 10M vectors       | Better hardware may be required     | Significantly higher cost |

## Assumptions

* The application runs on an existing local machine or server.
* The embedding model used is `all-MiniLM-L6-v2` with 384 dimensions.
* Hardware costs are not included because the system is assumed to run on an existing machine.
* The comparison is based on the general behavior of managed vector databases.

## ChromaDB Advantages

* Free and open-source.
* Easy to run locally.
* No separate database subscription cost.
* Suitable for small and medium-sized datasets.
* Data remains within the local system.

## ChromaDB Limitations

* Backups must be managed manually.
* Large datasets may require better hardware.
* Scaling to very large datasets can be more difficult.

## Conclusion

For this assignment, ChromaDB was selected because it satisfies the requirement of using a low-cost vector database. The retrieval evaluation achieved a Hit Rate, Recall, MRR, and nDCG score of 1.0 on the evaluation dataset, indicating that the local vector database was able to retrieve relevant information effectively. Although larger datasets may require additional hardware resources, ChromaDB provides a practical and cost-effective solution for small and medium-scale RAG applications.
