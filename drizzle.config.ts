import { defineConfig } from "drizzle-kit";

export default defineConfig({
  dialect: "sqlite",
  schema: "./packages/main/src/db/schema.ts",
  out: "./drizzle",
  dbCredentials: {
     url: process.env.DATABASE_URL ?? "file:./apex.dev.db"
  }
});


