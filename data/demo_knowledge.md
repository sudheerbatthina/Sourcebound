# Company Knowledge Base

## Company FAQ

### What does your company do?
We help organizations unlock the value locked in their documents and internal knowledge. Our platform lets teams ask questions in plain language and get instant, accurate answers sourced directly from their own files, wikis, and data systems — no more digging through folders or asking colleagues where things live.

### How do we work with clients?
We follow a structured onboarding process:
1. **Discovery call** — understand your current knowledge gaps, document formats, and team size.
2. **Pilot setup** — load a sample of your documents and configure the workspace for your team.
3. **Rollout** — deploy to your full team with training, access controls, and integrations in place.
4. **Ongoing support** — monthly check-ins, usage reviews, and continuous improvement.

### What types of documents does the system support?
The platform supports PDFs, Word documents, Markdown files, spreadsheets, and plain text. Integration connectors are available for Google Drive, Notion, Confluence, and Slack.

### Is our data secure?
Yes. All data is stored in tenant-isolated collections. Documents are never shared between organizations. The system supports SSO via Google OAuth, role-based access control, and optional data residency restrictions. No document content is used to train external AI models.

---

## Pricing Model

### How does pricing work?
Pricing is structured in three tiers:

| Plan | Monthly Price | Included |
|------|--------------|---------|
| **Starter** | $49/month | 3 users, 100 queries/month, 1 GB storage |
| **Pro** | $149/month | 20 users, 1,000 queries/month, 10 GB storage |
| **Enterprise** | Custom | Unlimited users, custom query volume, dedicated support |

All plans include a 14-day free trial. Annual billing saves 20%.

### Are there overage charges?
Pro plan users can purchase query packs ($29 per 500 additional queries). Enterprise plans include custom overage agreements negotiated upfront.

### What payment methods do you accept?
We accept all major credit cards, ACH bank transfer (US only), and invoicing for Enterprise accounts. Payment is processed securely via Stripe.

### Can I switch plans?
Yes. You can upgrade or downgrade at any time. Upgrades take effect immediately; downgrades apply at the start of the next billing cycle.

---

## HR Policy Excerpt

### Leave Policy

**Annual Leave**
Full-time employees accrue 15 days of paid annual leave per year, pro-rated in the first year. Leave must be approved by your manager at least 5 business days in advance for periods up to 5 days, and 3 weeks in advance for periods of 6 or more days.

**Sick Leave**
Employees receive 10 days of paid sick leave per year. Sick leave does not roll over. A doctor's note is required for absences of 3 or more consecutive days.

**Parental Leave**
Primary caregivers receive 16 weeks of fully paid parental leave. Secondary caregivers receive 4 weeks. Leave can begin up to 2 weeks before the expected birth or adoption date.

**Public Holidays**
The company observes all federal public holidays plus 2 floating days that employees may use at their discretion.

---

## Remote Work Policy

### Are employees allowed to work remotely?
Yes. We operate as a hybrid-first company. Employees may work remotely up to 4 days per week. There is no requirement to be in-office on a specific day, but teams are encouraged to align on at least one shared in-office day per week for collaboration.

### Who is eligible for fully remote work?
Employees who have been with the company for 6 or more months, are in good standing, and whose role does not require on-site physical presence may apply for fully remote status. Approval is at the discretion of the department head and HR.

### What equipment is provided for remote workers?
Remote employees receive a laptop, monitor, keyboard, and mouse on their first day. A one-time home office stipend of $500 is available to all employees working remotely more than 3 days per week.

### Are there restrictions on remote work locations?
Employees may work from anywhere within their country of employment. Working from abroad for more than 30 consecutive days requires prior approval from HR and Legal due to tax and employment law implications.

---

## Product Specification Excerpt

### Core Features

**Semantic Search**
The platform uses vector-based semantic search powered by OpenAI embeddings. Queries are matched to the most relevant document passages even when the exact keywords are not present. Results are ranked by relevance score.

**Streaming Answers**
Answers are streamed token-by-token for a fast, conversational feel. Each answer includes source citations linking back to the specific document page that informed the response.

**Multi-Tenant Workspaces**
Each organization gets an isolated workspace with separate document collections, user management, and usage quotas. Tenant data is stored in separate Chroma collections and cannot bleed between accounts.

**Role-Based Access Control**
Three roles are supported: Admin, Member, and Viewer. Admins manage documents and users. Members can query and upload. Viewers can only query. Access levels can also be set per document (public, restricted, admin-only).

**Chat History**
All conversations are persisted per user and per chat session. Users can rename, export, and delete conversations. Chat history is stored in SQLite and can be cleared at any time.

### Upcoming Features (Roadmap)
- Slack and Microsoft Teams integration for in-channel Q&A
- Scheduled document sync from Google Drive and Confluence
- Answer confidence scoring with source grounding percentage
- Custom branding and white-label deployment options
