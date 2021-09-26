from sqlalchemy.schema import DDL

valid_geom_function = DDL("""
CREATE OR REPLACE FUNCTION validate_geom()
  RETURNS trigger AS
$BODY$
  BEGIN
      NEW.geom = ST_MAKEVALID(NEW.geom);
      RETURN NEW;
    EXCEPTION WHEN OTHERS THEN
      NEW.ignore = true;
      RETURN NEW;
END;
$BODY$

LANGUAGE plpgsql VOLATILE -- Says the function is implemented in the plpgsql language; VOLATILE says the function has side effects.
COST 100; -- Estimated execution cost of the function.
""")

valid_geom_trigger = DDL("""
CREATE TRIGGER image_inserted
  BEFORE INSERT OR UPDATE
  ON images
  FOR EACH ROW
EXECUTE PROCEDURE validate_geom();
""")

valid_point_function = DDL("""
CREATE OR REPLACE FUNCTION validate_points()
  RETURNS trigger AS
$BODY$
BEGIN
 IF (SELECT COUNT(*)
	 FROM MEASURES
	 WHERE pointid = NEW.pointid AND "measureIgnore" = False) < 2
 THEN
   UPDATE points
     SET "pointIgnore" = True
	 WHERE points.id = NEW.pointid;
 ELSE
   UPDATE points
   SET "pointIgnore" = False
   WHERE points.id = NEW.pointid;
 END IF;

 RETURN NEW;
END;
$BODY$

LANGUAGE plpgsql VOLATILE -- Says the function is implemented in the plpgsql language; VOLATILE says the function has side effects.
COST 100; -- Estimated execution cost of the function.
""")

valid_point_trigger = DDL("""
CREATE TRIGGER active_measure_changes
  AFTER UPDATE
  ON measures
  FOR EACH ROW
EXECUTE PROCEDURE validate_points();
""")

ignore_image_function = DDL("""
CREATE OR REPLACE FUNCTION ignore_image()
  RETURNS trigger AS
$BODY$
BEGIN
 IF NEW.ignore
 THEN
   UPDATE measures
     SET "measureIgnore" = True
     WHERE measures.serialnumber = NEW.serial;
 END IF;

 RETURN NEW;
END;
$BODY$

LANGUAGE plpgsql VOLATILE -- Says the function is implemented in the plpgsql language; VOLATILE says the function has side effects.
COST 100; -- Estimated execution cost of the function.
""")

ignore_image_trigger = DDL("""
CREATE TRIGGER image_ignored
  AFTER UPDATE
  ON images
  FOR EACH ROW
EXECUTE PROCEDURE ignore_image();
""")


# several funcs and an operator needed to get json diff working. 
jsonb_delete_func = DDL("""
SET search_path = 'public';

CREATE OR REPLACE FUNCTION jsonb_delete_left(a jsonb, b text) 
RETURNS jsonb AS 
$BODY$
    SELECT COALESCE(    	
        (
            SELECT ('{' || string_agg(to_json(key) || ':' || value, ',') || '}')
            FROM jsonb_each(a)
            WHERE key <> b
        )
    , '{}')::jsonb;
$BODY$
LANGUAGE sql IMMUTABLE STRICT;
COMMENT ON FUNCTION jsonb_delete_left(jsonb, text) IS 'delete key in second argument from first argument';

CREATE OPERATOR - ( PROCEDURE = jsonb_delete_left, LEFTARG = jsonb, RIGHTARG = text);
COMMENT ON OPERATOR - (jsonb, text) IS 'delete key from left operand';

--

CREATE OR REPLACE FUNCTION jsonb_delete_left(a jsonb, b text[]) 
RETURNS jsonb AS 
$BODY$
    SELECT COALESCE(    	
        (
            SELECT ('{' || string_agg(to_json(key) || ':' || value, ',') || '}')
            FROM jsonb_each(a)
            WHERE key <> ALL(b)
        )
    , '{}')::jsonb;
$BODY$
LANGUAGE sql IMMUTABLE STRICT;
COMMENT ON FUNCTION jsonb_delete_left(jsonb, text[]) IS 'delete keys in second argument from first argument';

CREATE OPERATOR - ( PROCEDURE = jsonb_delete_left, LEFTARG = jsonb, RIGHTARG = text[]);
COMMENT ON OPERATOR - (jsonb, text[]) IS 'delete keys from left operand';

--

CREATE OR REPLACE FUNCTION jsonb_delete_left(a jsonb, b jsonb) 
RETURNS jsonb AS 
$BODY$
    SELECT COALESCE(    	
        (
            SELECT ('{' || string_agg(to_json(key) || ':' || value, ',') || '}')
            FROM jsonb_each(a)
            WHERE NOT ('{' || to_json(key) || ':' || value || '}')::jsonb <@ b
        )
    , '{}')::jsonb;
$BODY$
LANGUAGE sql IMMUTABLE STRICT;
COMMENT ON FUNCTION jsonb_delete_left(jsonb, jsonb) IS 'delete matching pairs in second argument from first argument';

CREATE OPERATOR - ( PROCEDURE = jsonb_delete_left, LEFTARG = jsonb, RIGHTARG = jsonb);
COMMENT ON OPERATOR - (jsonb, jsonb) IS 'delete matching pairs from left operand';
""")


def generate_history_triggers(table):
  tablename = table.__tablename__

  history_update_function = DDL(f"""
  CREATE OR REPLACE FUNCTION {tablename}_history_update()
  RETURNS TRIGGER AS $$
  DECLARE
    js_new jsonb := row_to_json(NEW)::jsonb;
    js_old jsonb := row_to_json(OLD)::jsonb;
  BEGIN
    INSERT INTO {tablename}_history(fk, "eventTime", "executedBy", event, before, after)
      VALUES((js_old->>'id')::int, CURRENT_TIMESTAMP, SESSION_USER, 'update', js_old - js_new, js_new - js_old);
    RETURN NEW;
  END;

  $$ LANGUAGE plpgsql;
  """)

  history_insert_function = DDL(f"""
  CREATE OR REPLACE FUNCTION {tablename}_history_insert()
  RETURNS TRIGGER AS $$
  DECLARE 
    js_new jsonb := row_to_json(NEW)::jsonb;
  BEGIN
    INSERT INTO {tablename}_history(fk, "eventTime", "executedBy", event, after)
       VALUES((js_new->>'id')::int, CURRENT_TIMESTAMP, SESSION_USER, 'insert', js_new);
    RETURN NEW;
  END;
  $$ LANGUAGE plpgsql;
  """)

  history_delete_function = DDL(f"""
  CREATE OR REPLACE FUNCTION {tablename}_history_delete()
  RETURNS TRIGGER AS $$
  DECLARE
    js_old jsonb := row_to_json(OLD)::jsonb;
  BEGIN
    INSERT INTO {tablename}_history(fk, "eventTime", "executedBy", event, before)
       VALUES((js_old->>'id')::int, CURRENT_TIMESTAMP, SESSION_USER, 'delete', js_old);
    RETURN NEW;
  END;
  $$ LANGUAGE plpgsql;
  """)

  history_insert_trigger = DDL(f"""
  CREATE TRIGGER {tablename}_history_insert AFTER INSERT ON {tablename}
    FOR EACH ROW EXECUTE PROCEDURE {tablename}_history_insert();
  """)

  history_delete_trigger = DDL(f"""
  CREATE TRIGGER {tablename}_history_delete AFTER DELETE ON {tablename}
    FOR EACH ROW EXECUTE PROCEDURE {tablename}_history_delete();
  """)

  history_update_trigger = DDL(f"""
  CREATE TRIGGER {tablename}_history_update AFTER UPDATE ON {tablename}
    FOR EACH ROW EXECUTE PROCEDURE {tablename}_history_update();
  """)

  return history_update_function, history_insert_function, history_delete_function, history_insert_trigger, history_delete_trigger, history_update_trigger 