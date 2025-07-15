add_executable(train_faiss_ivfflat train_faiss_ivfflat.cpp)
target_link_libraries(train_faiss_ivfflat
    PRIVATE
    ${PROJECT_NAME}
)

add_executable(extract_netvlad_desc extract_netvlad_desc.cpp)
target_link_libraries(extract_netvlad_desc
    PRIVATE
    ${PROJECT_NAME}
    pthread
)

install(TARGETS train_faiss_ivfflat extract_netvlad_desc DESTINATION bin)


