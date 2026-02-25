add_executable(train_faiss_ivfflat train_faiss_ivfflat.cpp)
target_link_libraries(train_faiss_ivfflat
    PRIVATE
    ${PROJECT_NAME}
)

install(TARGETS train_faiss_ivfflat DESTINATION bin)


