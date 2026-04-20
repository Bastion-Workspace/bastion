-- Row Level Security for Agent Factory tier-1 tables (user_id column).
-- Ensure columns referenced by policies exist on older DBs.
ALTER TABLE custom_playbooks ADD COLUMN IF NOT EXISTS is_builtin BOOLEAN DEFAULT false;
ALTER TABLE agent_profiles ADD COLUMN IF NOT EXISTS is_builtin BOOLEAN DEFAULT false;

-- Uses app.current_user_id / app.current_user_role (see database_manager_service.execute_query).
-- Rollback: run the DROP POLICY / DISABLE blocks at the end of this file.

-- ---------------------------------------------------------------------------
-- Helper expressions (documented only; inlined in policies for portability)
-- current_setting('app.current_user_id', true)::varchar
-- current_setting('app.current_user_role', true) = 'admin'
-- ---------------------------------------------------------------------------

-- agent_execution_log
ALTER TABLE agent_execution_log ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS agent_execution_log_all ON agent_execution_log;
CREATE POLICY agent_execution_log_all ON agent_execution_log FOR ALL
  USING (
    current_setting('app.current_user_role', true) = 'admin'
    OR user_id = current_setting('app.current_user_id', true)::varchar
  )
  WITH CHECK (
    current_setting('app.current_user_role', true) = 'admin'
    OR user_id = current_setting('app.current_user_id', true)::varchar
  );

-- agent_discoveries
ALTER TABLE agent_discoveries ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS agent_discoveries_all ON agent_discoveries;
CREATE POLICY agent_discoveries_all ON agent_discoveries FOR ALL
  USING (
    current_setting('app.current_user_role', true) = 'admin'
    OR user_id = current_setting('app.current_user_id', true)::varchar
  )
  WITH CHECK (
    current_setting('app.current_user_role', true) = 'admin'
    OR user_id = current_setting('app.current_user_id', true)::varchar
  );

-- agent_schedules
ALTER TABLE agent_schedules ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS agent_schedules_all ON agent_schedules;
CREATE POLICY agent_schedules_all ON agent_schedules FOR ALL
  USING (
    current_setting('app.current_user_role', true) = 'admin'
    OR user_id = current_setting('app.current_user_id', true)::varchar
  )
  WITH CHECK (
    current_setting('app.current_user_role', true) = 'admin'
    OR user_id = current_setting('app.current_user_id', true)::varchar
  );

-- agent_budgets (runner may be sharee; budget row owned by profile owner)
ALTER TABLE agent_budgets ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS agent_budgets_all ON agent_budgets;
CREATE POLICY agent_budgets_all ON agent_budgets FOR ALL
  USING (
    current_setting('app.current_user_role', true) = 'admin'
    OR user_id = current_setting('app.current_user_id', true)::varchar
    OR EXISTS (
      SELECT 1 FROM agent_profiles p
      WHERE p.id = agent_budgets.agent_profile_id
        AND (
          p.user_id = current_setting('app.current_user_id', true)::varchar
          OR EXISTS (
            SELECT 1 FROM agent_artifact_shares sh
            WHERE sh.artifact_type = 'agent_profile'
              AND sh.artifact_id = p.id
              AND sh.shared_with_user_id = current_setting('app.current_user_id', true)::varchar
          )
        )
    )
  )
  WITH CHECK (
    current_setting('app.current_user_role', true) = 'admin'
    OR user_id = current_setting('app.current_user_id', true)::varchar
    OR EXISTS (
      SELECT 1 FROM agent_profiles p
      WHERE p.id = agent_budgets.agent_profile_id
        AND (
          p.user_id = current_setting('app.current_user_id', true)::varchar
          OR EXISTS (
            SELECT 1 FROM agent_artifact_shares sh
            WHERE sh.artifact_type = 'agent_profile'
              AND sh.artifact_id = p.id
              AND sh.shared_with_user_id = current_setting('app.current_user_id', true)::varchar
          )
        )
    )
  );

-- agent_approval_queue
ALTER TABLE agent_approval_queue ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS agent_approval_queue_all ON agent_approval_queue;
CREATE POLICY agent_approval_queue_all ON agent_approval_queue FOR ALL
  USING (
    current_setting('app.current_user_role', true) = 'admin'
    OR user_id = current_setting('app.current_user_id', true)::varchar
  )
  WITH CHECK (
    current_setting('app.current_user_role', true) = 'admin'
    OR user_id = current_setting('app.current_user_id', true)::varchar
  );

-- agent_memory
ALTER TABLE agent_memory ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS agent_memory_all ON agent_memory;
CREATE POLICY agent_memory_all ON agent_memory FOR ALL
  USING (
    current_setting('app.current_user_role', true) = 'admin'
    OR user_id = current_setting('app.current_user_id', true)::varchar
  )
  WITH CHECK (
    current_setting('app.current_user_role', true) = 'admin'
    OR user_id = current_setting('app.current_user_id', true)::varchar
  );

-- agent_lines
ALTER TABLE agent_lines ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS agent_lines_all ON agent_lines;
CREATE POLICY agent_lines_all ON agent_lines FOR ALL
  USING (
    current_setting('app.current_user_role', true) = 'admin'
    OR user_id = current_setting('app.current_user_id', true)::varchar
  )
  WITH CHECK (
    current_setting('app.current_user_role', true) = 'admin'
    OR user_id = current_setting('app.current_user_id', true)::varchar
  );

-- agent_factory_sidebar_categories
ALTER TABLE agent_factory_sidebar_categories ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS agent_factory_sidebar_categories_all ON agent_factory_sidebar_categories;
CREATE POLICY agent_factory_sidebar_categories_all ON agent_factory_sidebar_categories FOR ALL
  USING (
    current_setting('app.current_user_role', true) = 'admin'
    OR user_id = current_setting('app.current_user_id', true)::varchar
  )
  WITH CHECK (
    current_setting('app.current_user_role', true) = 'admin'
    OR user_id = current_setting('app.current_user_id', true)::varchar
  );

-- agent_profiles
ALTER TABLE agent_profiles ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS agent_profiles_select ON agent_profiles;
CREATE POLICY agent_profiles_select ON agent_profiles FOR SELECT USING (
  current_setting('app.current_user_role', true) = 'admin'
  OR user_id = current_setting('app.current_user_id', true)::varchar
  OR EXISTS (
    SELECT 1 FROM agent_artifact_shares sh
    WHERE sh.artifact_type = 'agent_profile'
      AND sh.artifact_id = agent_profiles.id
      AND sh.shared_with_user_id = current_setting('app.current_user_id', true)::varchar
  )
);
DROP POLICY IF EXISTS agent_profiles_insert ON agent_profiles;
CREATE POLICY agent_profiles_insert ON agent_profiles FOR INSERT WITH CHECK (
  current_setting('app.current_user_role', true) = 'admin'
  OR user_id = current_setting('app.current_user_id', true)::varchar
);
DROP POLICY IF EXISTS agent_profiles_update ON agent_profiles;
CREATE POLICY agent_profiles_update ON agent_profiles FOR UPDATE USING (
  current_setting('app.current_user_role', true) = 'admin'
  OR user_id = current_setting('app.current_user_id', true)::varchar
) WITH CHECK (
  current_setting('app.current_user_role', true) = 'admin'
  OR user_id = current_setting('app.current_user_id', true)::varchar
);
DROP POLICY IF EXISTS agent_profiles_delete ON agent_profiles;
CREATE POLICY agent_profiles_delete ON agent_profiles FOR DELETE USING (
  current_setting('app.current_user_role', true) = 'admin'
  OR user_id = current_setting('app.current_user_id', true)::varchar
);

-- custom_playbooks
ALTER TABLE custom_playbooks ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS custom_playbooks_select ON custom_playbooks;
CREATE POLICY custom_playbooks_select ON custom_playbooks FOR SELECT USING (
  current_setting('app.current_user_role', true) = 'admin'
  OR COALESCE(is_builtin, false) = true
  OR user_id IS NULL
  OR user_id = current_setting('app.current_user_id', true)::varchar
  OR EXISTS (
    SELECT 1 FROM agent_artifact_shares sh
    WHERE sh.artifact_type = 'playbook'
      AND sh.artifact_id = custom_playbooks.id
      AND sh.shared_with_user_id = current_setting('app.current_user_id', true)::varchar
  )
);
DROP POLICY IF EXISTS custom_playbooks_insert ON custom_playbooks;
CREATE POLICY custom_playbooks_insert ON custom_playbooks FOR INSERT WITH CHECK (
  current_setting('app.current_user_role', true) = 'admin'
  OR user_id = current_setting('app.current_user_id', true)::varchar
);
DROP POLICY IF EXISTS custom_playbooks_update ON custom_playbooks;
CREATE POLICY custom_playbooks_update ON custom_playbooks FOR UPDATE USING (
  current_setting('app.current_user_role', true) = 'admin'
  OR user_id = current_setting('app.current_user_id', true)::varchar
) WITH CHECK (
  current_setting('app.current_user_role', true) = 'admin'
  OR user_id = current_setting('app.current_user_id', true)::varchar
);
DROP POLICY IF EXISTS custom_playbooks_delete ON custom_playbooks;
CREATE POLICY custom_playbooks_delete ON custom_playbooks FOR DELETE USING (
  current_setting('app.current_user_role', true) = 'admin'
  OR user_id = current_setting('app.current_user_id', true)::varchar
);

-- agent_skills (nullable user_id for built-ins)
DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'agent_skills') THEN
    ALTER TABLE agent_skills ENABLE ROW LEVEL SECURITY;
    DROP POLICY IF EXISTS agent_skills_select ON agent_skills;
    CREATE POLICY agent_skills_select ON agent_skills FOR SELECT USING (
      current_setting('app.current_user_role', true) = 'admin'
      OR user_id IS NULL
      OR user_id = current_setting('app.current_user_id', true)::varchar
      OR EXISTS (
        SELECT 1 FROM agent_artifact_shares sh
        WHERE sh.artifact_type = 'skill'
          AND sh.artifact_id = agent_skills.id
          AND sh.shared_with_user_id = current_setting('app.current_user_id', true)::varchar
      )
    );
    DROP POLICY IF EXISTS agent_skills_insert ON agent_skills;
    CREATE POLICY agent_skills_insert ON agent_skills FOR INSERT WITH CHECK (
      current_setting('app.current_user_role', true) = 'admin'
      OR user_id = current_setting('app.current_user_id', true)::varchar
    );
    DROP POLICY IF EXISTS agent_skills_update ON agent_skills;
    CREATE POLICY agent_skills_update ON agent_skills FOR UPDATE USING (
      current_setting('app.current_user_role', true) = 'admin'
      OR user_id = current_setting('app.current_user_id', true)::varchar
    ) WITH CHECK (
      current_setting('app.current_user_role', true) = 'admin'
      OR user_id = current_setting('app.current_user_id', true)::varchar
    );
    DROP POLICY IF EXISTS agent_skills_delete ON agent_skills;
    CREATE POLICY agent_skills_delete ON agent_skills FOR DELETE USING (
      current_setting('app.current_user_role', true) = 'admin'
      OR user_id = current_setting('app.current_user_id', true)::varchar
    );
  END IF;
END $$;

-- Watch tables (migrations)
DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'agent_email_watches') THEN
    ALTER TABLE agent_email_watches ENABLE ROW LEVEL SECURITY;
    DROP POLICY IF EXISTS agent_email_watches_all ON agent_email_watches;
    CREATE POLICY agent_email_watches_all ON agent_email_watches FOR ALL
      USING (
        current_setting('app.current_user_role', true) = 'admin'
        OR user_id = current_setting('app.current_user_id', true)::varchar
      )
      WITH CHECK (
        current_setting('app.current_user_role', true) = 'admin'
        OR user_id = current_setting('app.current_user_id', true)::varchar
      );
  END IF;
  IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'agent_folder_watches') THEN
    ALTER TABLE agent_folder_watches ENABLE ROW LEVEL SECURITY;
    DROP POLICY IF EXISTS agent_folder_watches_all ON agent_folder_watches;
    CREATE POLICY agent_folder_watches_all ON agent_folder_watches FOR ALL
      USING (
        current_setting('app.current_user_role', true) = 'admin'
        OR user_id = current_setting('app.current_user_id', true)::varchar
      )
      WITH CHECK (
        current_setting('app.current_user_role', true) = 'admin'
        OR user_id = current_setting('app.current_user_id', true)::varchar
      );
  END IF;
  IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'agent_conversation_watches') THEN
    ALTER TABLE agent_conversation_watches ENABLE ROW LEVEL SECURITY;
    DROP POLICY IF EXISTS agent_conversation_watches_all ON agent_conversation_watches;
    CREATE POLICY agent_conversation_watches_all ON agent_conversation_watches FOR ALL
      USING (
        current_setting('app.current_user_role', true) = 'admin'
        OR user_id = current_setting('app.current_user_id', true)::varchar
      )
      WITH CHECK (
        current_setting('app.current_user_role', true) = 'admin'
        OR user_id = current_setting('app.current_user_id', true)::varchar
      );
  END IF;
  IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'agent_line_watches') THEN
    ALTER TABLE agent_line_watches ENABLE ROW LEVEL SECURITY;
    DROP POLICY IF EXISTS agent_line_watches_all ON agent_line_watches;
    CREATE POLICY agent_line_watches_all ON agent_line_watches FOR ALL
      USING (
        current_setting('app.current_user_role', true) = 'admin'
        OR user_id = current_setting('app.current_user_id', true)::varchar
      )
      WITH CHECK (
        current_setting('app.current_user_role', true) = 'admin'
        OR user_id = current_setting('app.current_user_id', true)::varchar
      );
  END IF;
END $$;
