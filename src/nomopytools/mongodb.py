from pymongo import MongoClient
from pymongo.database import Database, Collection
from pymongo.results import InsertOneResult, InsertManyResult, UpdateResult
from pymongo.cursor import Cursor


class MDB:
    """Start mongodb server:
    mongod --config /opt/homebrew/etc/mongod.conf --fork
    Stop mongodb server:
    ps aux | grep -v grep | grep mongod
    """

    def __init__(self, inst: str = "mongodb://localhost:27017/") -> None:
        self.inst = inst
        self.client = MongoClient(inst)

    def get_db(self, db: str | Database) -> Database:
        return db if isinstance(db, Database) else self.client[db]

    def get_collection(
        self, collection: str | Collection, db: str | Database | None = None
    ) -> Collection:
        if isinstance(collection, Collection):
            return collection
        elif db:
            return self.get_db(db)[collection]
        raise Exception("If collection is str, db needs to be given.")

    def create_db(self, db_name: str) -> Database:
        return self.get_db(db_name)

    def create_collection(
        self, collection_name: str | Collection, db: str | Database
    ) -> Collection:
        return self.get_db(db)[collection_name]

    def check_db_exists(self, db: str) -> bool:
        return db in self.client.list_database_names()

    def check_collection_exists(self, collection: str) -> bool:
        return collection in self.client.list_collections()

    def insert(
        self,
        doc: dict | list[dict],
        collection: str | Collection,
        ordered: bool = True,
        db: str | Database | None = None,
    ) -> InsertOneResult | InsertManyResult:
        return (
            self.get_collection(collection, db).insert_one(doc)
            if isinstance(doc, dict)
            else self.get_collection(collection, db).insert_many(doc)
        )

    def update(
        self,
        filter: dict,
        update: dict | list[dict],
        collection: str | Collection,
        upsert: bool = True,
        db: str | Database | None = None,
    ) -> UpdateResult:
        return (
            self.get_collection(collection, db).update_one(
                filter=filter, update={"$set": update}, upsert=upsert
            )
            if isinstance(update, dict)
            else self.get_collection(collection, db).update_many(
                filter=filter, update={"$set": update}, upsert=upsert
            )
        )

    def query(
        self,
        collection: str | Collection,
        db: str | Database | None = None,
        filter: dict | None = None,
        projection: dict | list | None = None,
    ) -> list[dict]:
        return list(
            self.find(
                filter=filter, collection=collection, db=db, projection=projection
            )
        )

    def find(
        self,
        collection: str | Collection,
        db: str | Database | None = None,
        filter: dict | None = None,
        projection: dict | list | None = None,
    ) -> Cursor:
        return self.get_collection(collection, db).find(
            filter=filter, projection=projection
        )
