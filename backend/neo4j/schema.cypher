CREATE CONSTRAINT repository_id IF NOT EXISTS
FOR (r:Repository)
REQUIRE r.id IS UNIQUE;

CREATE CONSTRAINT repository_name IF NOT EXISTS
FOR (r:Repository)
REQUIRE r.name IS UNIQUE;

CREATE CONSTRAINT folder_id IF NOT EXISTS
FOR (f:Folder)
REQUIRE f.id IS UNIQUE;

CREATE CONSTRAINT document_id IF NOT EXISTS
FOR (d:Document)
REQUIRE d.id IS UNIQUE;

CREATE CONSTRAINT chunk_id IF NOT EXISTS
FOR (c:Chunk)
REQUIRE c.id IS UNIQUE;

CREATE CONSTRAINT customer_name IF NOT EXISTS
FOR (c:Customer)
REQUIRE c.name IS UNIQUE;

CREATE CONSTRAINT vendor_name IF NOT EXISTS
FOR (v:Vendor)
REQUIRE v.name IS UNIQUE;

CREATE CONSTRAINT sla_id IF NOT EXISTS
FOR (s:SLA)
REQUIRE s.id IS UNIQUE;

CREATE CONSTRAINT incident_id IF NOT EXISTS
FOR (i:Incident)
REQUIRE i.id IS UNIQUE;

CREATE CONSTRAINT team_name IF NOT EXISTS
FOR (t:Team)
REQUIRE t.name IS UNIQUE;

CREATE FULLTEXT INDEX datameta_repository_fulltext IF NOT EXISTS
FOR (r:Repository)
ON EACH [r.name, r.title, r.summary, r.metadata_text];

CREATE FULLTEXT INDEX datameta_folder_fulltext IF NOT EXISTS
FOR (f:Folder)
ON EACH [f.repository, f.folder, f.title, f.summary, f.metadata_text];

CREATE FULLTEXT INDEX datameta_document_fulltext IF NOT EXISTS
FOR (d:Document)
ON EACH [d.repository, d.folder, d.path, d.type, d.title, d.summary, d.metadata_text];

CREATE FULLTEXT INDEX datameta_chunk_fulltext IF NOT EXISTS
FOR (c:Chunk)
ON EACH [c.path, c.heading, c.text];

CREATE VECTOR INDEX datameta_repository_embedding IF NOT EXISTS
FOR (r:Repository)
ON r.embedding
OPTIONS {
  indexConfig: {
    `vector.dimensions`: 3072,
    `vector.similarity_function`: 'cosine'
  }
};

CREATE VECTOR INDEX datameta_folder_embedding IF NOT EXISTS
FOR (f:Folder)
ON f.embedding
OPTIONS {
  indexConfig: {
    `vector.dimensions`: 3072,
    `vector.similarity_function`: 'cosine'
  }
};

CREATE VECTOR INDEX datameta_document_embedding IF NOT EXISTS
FOR (d:Document)
ON d.embedding
OPTIONS {
  indexConfig: {
    `vector.dimensions`: 3072,
    `vector.similarity_function`: 'cosine'
  }
};
