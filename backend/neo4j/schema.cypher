CREATE CONSTRAINT document_id IF NOT EXISTS
FOR (d:Document)
REQUIRE d.id IS UNIQUE;

CREATE CONSTRAINT definition_id IF NOT EXISTS
FOR (d:Definition)
REQUIRE d.id IS UNIQUE;

CREATE CONSTRAINT runbook_id IF NOT EXISTS
FOR (r:Runbook)
REQUIRE r.id IS UNIQUE;

CREATE CONSTRAINT table_name IF NOT EXISTS
FOR (t:Table)
REQUIRE t.name IS UNIQUE;

CREATE CONSTRAINT column_key IF NOT EXISTS
FOR (c:Column)
REQUIRE c.key IS UNIQUE;

CREATE FULLTEXT INDEX datameta_fulltext IF NOT EXISTS
FOR (n:Document|Definition|Runbook|Policy|DataQualityFlag)
ON EACH [n.title, n.summary, n.body, n.entity, n.scope, n.team];

CREATE VECTOR INDEX datameta_embedding IF NOT EXISTS
FOR (n:Document)
ON n.embedding
OPTIONS {
  indexConfig: {
    `vector.dimensions`: 3072,
    `vector.similarity_function`: 'cosine'
  }
};
